import Foundation
import AVFoundation
import Speech
import CryptoKit

struct Segment: Codable, Sendable {
    let start: Double
    let end: Double
    let text: String
    var units: [Unit] = []
}
struct Unit: Codable, Sendable {
    let start: Double?
    let end: Double?
    let text: String
}

@main
struct Transcribe {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments == ["--check-model"] || arguments == ["--prepare-model"] {
            let ready = await modelCommand(prepare: arguments[0] == "--prepare-model")
            exit(ready ? 0 : 1)
        }
        do {
            try await run()
        } catch {
            let error = error as NSError
            fputs("ERROR \(error.domain) code=\(error.code): \(error.localizedDescription)\n\(error.userInfo)\n", stderr)
            exit(1)
        }
    }

    /// Checking never reserves or downloads assets. Only the explicit prepare
    /// command may request Apple's system-managed Japanese model installation.
    static func modelCommand(prepare: Bool) async -> Bool {
        let supportedLocales = await SpeechTranscriber.supportedLocales
        let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "ja-JP"))
        var report: [String: Any] = [
            "command": prepare ? "prepare-model" : "check-model",
            "requested_locale": "ja-JP",
            "locale": locale?.identifier as Any? ?? NSNull(),
            "supported_locales": supportedLocales.map(\.identifier).sorted(),
            "supported": SpeechTranscriber.isAvailable && locale != nil,
            "installed": false,
            "status": "unavailable",
            "installation_requested": false,
            "audio_uploaded": false,
            "exact_model_revision": NSNull(),
            "download_bytes": NSNull(),
            "model_source": "Apple system-managed SpeechTranscriber assets",
            "model_details_note": "Apple does not expose a fixed model revision or download byte count through this command."
        ]
        var ready = false
        if SpeechTranscriber.isAvailable, let locale {
            let installed = await SpeechTranscriber.installedLocales
            ready = installed.contains { $0.identifier == locale.identifier }
            report["installed"] = ready
            report["status"] = ready ? "ready" : "missing"
            if prepare && !ready {
                do {
                    let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [], attributeOptions: [.audioTimeRange])
                    // This is the only model download entry point. Transcription
                    // itself continues to fail when the model is not installed.
                    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                        report["installation_requested"] = true
                        fputs("Preparing Apple's Japanese SpeechTranscriber assets; download size is managed by macOS. No audio is sent.\n", stderr)
                        try await request.downloadAndInstall()
                    }
                    let after = await SpeechTranscriber.installedLocales
                    ready = after.contains { $0.identifier == locale.identifier }
                    report["installed"] = ready
                    report["status"] = ready ? "ready" : "missing"
                    if !ready {
                        report["error"] = "Installation did not make Japanese available. Inspect macOS connectivity or model status, then explicitly retry --prepare-model."
                    }
                } catch {
                    let error = error as NSError
                    report["status"] = "failed"
                    report["error"] = error.localizedDescription
                    report["error_domain"] = error.domain
                    report["error_code"] = error.code
                }
            }
        }
        do {
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        } catch {
            fputs("ERROR: Could not encode model status: \(error)\n", stderr)
            return false
        }
        return ready
    }

    static func run() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count == 2 || (arguments.count == 4 && arguments[2] == "--context-file") else {
            throw NSError(domain: "Transcribe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Usage: apple-transcribe input-audio output-directory [--context-file terms.json] | --check-model | --prepare-model"])
        }
        var contextMetadata: [String: Any]?
        var contextualStrings: [String]?
        if arguments.count == 4 {
            let contextURL = URL(fileURLWithPath: arguments[3]).standardizedFileURL
            let data = try Data(contentsOf: contextURL)
            let terms = try JSONDecoder().decode([String].self, from: data)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            guard terms.count <= 100, terms.allSatisfy({ !$0.isEmpty }) else {
                throw NSError(domain: "Transcribe", code: 6, userInfo: [NSLocalizedDescriptionKey: "Context must contain at most 100 nonempty strings"])
            }
            contextualStrings = terms
            contextMetadata = [
                "file": contextURL.path,
                "file_sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
                "terms": terms,
                "api": "AnalysisContext.contextualStrings[.general] / SpeechAnalyzer.setContext",
                "set_context_completed": false,
                "recognition_bias_verified": false,
                "note": "Official contextualStrings accuracy discussion names DictationTranscriber; setter completion does not establish an effect on SpeechTranscriber. No text substitution is performed.",
                "documentation": "https://developer.apple.com/documentation/speech/analysiscontext/contextualstrings"
            ]
        }
        guard SpeechTranscriber.isAvailable,
              let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "ja-JP")) else {
            throw NSError(domain: "Transcribe", code: 2, userInfo: [NSLocalizedDescriptionKey: "Japanese SpeechTranscriber is unavailable"])
        }
        let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [], attributeOptions: [.audioTimeRange])
        let installed = await SpeechTranscriber.installedLocales
        guard installed.contains(where: { $0.identifier == locale.identifier }) else {
            throw NSError(domain: "Transcribe", code: 3, userInfo: [NSLocalizedDescriptionKey: "Japanese model is not installed; no download was requested"])
        }
        let source = URL(fileURLWithPath: arguments[0])
        let output = URL(fileURLWithPath: arguments[1], isDirectory: true)
        guard !FileManager.default.fileExists(atPath: output.path) else { throw NSError(domain: "Transcribe", code: 5, userInfo: [NSLocalizedDescriptionKey: "Output already exists"]) }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let audio = try AVAudioFile(forReading: source)
        let started = Date()
        let duration = Double(audio.length) / audio.processingFormat.sampleRate
        print("Input: \(source.lastPathComponent), duration=\(duration), locale=\(locale.identifier)")
        fflush(stdout)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        if let terms = contextualStrings {
            let metadataURL = output.appendingPathComponent("context-metadata.json")
            try JSONSerialization.data(withJSONObject: contextMetadata!, options: [.prettyPrinted, .sortedKeys])
                .write(to: metadataURL, options: .atomic)
            let context = AnalysisContext()
            context.contextualStrings[.general] = terms
            do {
                try await analyzer.setContext(context)
            } catch {
                await analyzer.cancelAndFinishNow()
                throw error
            }
            contextMetadata?["set_context_completed"] = true
            try JSONSerialization.data(withJSONObject: contextMetadata!, options: [.prettyPrinted, .sortedKeys])
                .write(to: metadataURL, options: .atomic)
            print("Context: \(terms.count) terms submitted; recognition bias is unverified")
            fflush(stdout)
        }
        let collector = Task { () throws -> [Segment] in
            var segments: [Segment] = []
            for try await result in transcriber.results {
                let segment = Segment(start: result.range.start.seconds,
                                      end: CMTimeRangeGetEnd(result.range).seconds,
                                      text: String(result.text.characters),
                                      units: result.text.runs.map { run in
                                          let range = run.audioTimeRange
                                          return Unit(start: range?.start.seconds, end: range.map { CMTimeRangeGetEnd($0).seconds }, text: String(result.text[run.range].characters))
                                      })
                segments.append(segment)
                if segments.count == 1 || segments.count % 25 == 0 {
                    print("Progress: \(Int(segment.end)) / \(Int(duration)) seconds; \(segments.count) segments")
                    fflush(stdout)
                }
            }
            return segments
        }
        do {
            _ = try await analyzer.analyzeSequence(from: audio)
            try await analyzer.finalizeAndFinishThroughEndOfInput()
            let segments = try await collector.value
            guard !segments.isEmpty else {
                throw NSError(domain: "Transcribe", code: 4, userInfo: [NSLocalizedDescriptionKey: "No speech results were produced"])
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(segments).write(to: output.appendingPathComponent("transcript-segments.json"), options: .atomic)
            let text = segments.map(\.text).joined(separator: "\n")
            try text.write(to: output.appendingPathComponent("transcript.txt"), atomically: true, encoding: .utf8)
            let timestamped = segments.map { s in
                let seconds = Int(s.start)
                return String(format: "[%02d:%02d] ", seconds / 60, seconds % 60) + s.text
            }.joined(separator: "\n\n")
            let elapsed = Date().timeIntervalSince(started)
            let report = """
            # Apple SpeechTranscriber 文字起こし

            - 入力: \(source.lastPathComponent)
            - 日本語: \(locale.identifier)
            - 音声の長さ: \(duration) 秒
            - 処理時間: \(elapsed) 秒
            - 処理方式: Apple SpeechAnalyzer / SpeechTranscriber、端末内処理
            - 状態: 未校正の自動文字起こし。話者分離・要約は未実施。

            \(timestamped)
            """
            try report.write(to: output.appendingPathComponent("transcript.md"), atomically: true, encoding: .utf8)
            print("DONE: \(segments.count) segments, \(text.count) characters, elapsed=\(elapsed) seconds, last_segment_end=\(segments.last!.end)")
        } catch {
            await analyzer.cancelAndFinishNow()
            collector.cancel()
            throw error
        }
    }
}
