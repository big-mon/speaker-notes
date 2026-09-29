import CoreML
import CryptoKit
import Foundation

private struct CLIError: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

@main struct SileroVADFrames {
    static let sampleRate = 16_000
    static let chunkSize = 512
    static let contextSize = 64
    static let inputSchema = ["audio_input": [1, 576], "hidden_state": [1, 128], "cell_state": [1, 128]]
    static let outputSchema = ["vad_output": [1, 1, 1], "new_hidden_state": [1, 128], "new_cell_state": [1, 128]]

    static func main() {
        do { try run() }
        catch {
            FileHandle.standardError.write(Data("silero-vad-frames: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    static func run() throws {
        let started = ProcessInfo.processInfo.systemUptime
        let args = CommandLine.arguments
        guard args.count == 4 else {
            throw CLIError(message: "Usage: silero-vad-frames INPUT_FLOAT_WAV MODEL_BUNDLE NEW_OUTPUT_DIR")
        }
        let input = URL(fileURLWithPath: args[1]).standardizedFileURL
        let bundle = URL(fileURLWithPath: args[2]).standardizedFileURL
        let output = URL(fileURLWithPath: args[3]).standardizedFileURL
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw CLIError(message: "Output exists: \(output.path)")
        }
        let audio = try readInput(input)
        let metadataData = try Data(contentsOf: bundle.appendingPathComponent("metadata.json"))
        guard let metadataList = try JSONSerialization.jsonObject(with: metadataData) as? [[String: Any]],
              metadataList.count == 1, let metadata = metadataList.first,
              metadata["version"] as? String == "6.0.0" else {
            throw CLIError(message: "Expected Silero unified v6.0.0 metadata")
        }
        try checkMetadata(metadata["inputSchema"], expected: inputSchema)
        try checkMetadata(metadata["outputSchema"], expected: outputSchema)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var stage = "model_load"
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            let loadStart = ProcessInfo.processInfo.systemUptime
            let model = try MLModel(contentsOf: bundle, configuration: config)
            let loadSeconds = ProcessInfo.processInfo.systemUptime - loadStart
            stage = "model_schema"
            try writeJSON([
                "bundle_metadata": metadata,
                "actual_input_schema": describe(model.modelDescription.inputDescriptionsByName),
                "actual_output_schema": describe(model.modelDescription.outputDescriptionsByName),
                "actual_metadata": Dictionary(uniqueKeysWithValues: model.modelDescription.metadata.map {
                    ($0.key.rawValue, String(describing: $0.value))
                }),
            ], to: output.appendingPathComponent("model-description.json"))
            try checkActual(model.modelDescription.inputDescriptionsByName, expected: inputSchema)
            try checkActual(model.modelDescription.outputDescriptionsByName, expected: outputSchema)
            let paddingCount = (chunkSize - audio.samples.count % chunkSize) % chunkSize
            try writeJSON([
                "engine": "Silero VAD v6.0.0 32ms / direct Core ML",
                "model_bundle": bundle.path,
                "model_metadata_sha256": sha256(metadataData),
                "expected_model_version": "6.0.0",
                "model_schema_verified": true,
                "input_schema": inputSchema,
                "output_schema": outputSchema,
                "compute_units": "cpuOnly",
                "offline_model_loading": true,
                "sample_rate": sampleRate,
                "chunk_size_samples": chunkSize,
                "context_samples": contextSize,
                "initial_state": "hidden, cell and preceding context are zero",
                "state_policy": "hidden and cell carried between every frame; context is previous frame's last64 samples",
                "frame_tail_padding": "Zero padding for last model input only; saved frame end is clamped to actual source length",
                "final_frame_zero_padding_samples": paddingCount,
                "audio_mutation": "None; original PCM remains unchanged; no segmentation, deletion, splicing or ASR input padding",
                "probability_meaning": "Raw model output; not calibrated boundary confidence or verified non-speech",
                "macOS": ProcessInfo.processInfo.operatingSystemVersionString,
                "model_load_seconds": loadSeconds,
            ], to: output.appendingPathComponent("settings.json"))
            stage = "frame_inference"
            let inputArray = try zeroArray([1, chunkSize + contextSize])
            guard inputArray.strides.map(\.intValue) == [chunkSize + contextSize, 1] else {
                throw CLIError(message: "Unexpected allocated input array strides")
            }
            let inputPointer = inputArray.dataPointer.assumingMemoryBound(to: Float.self)
            var hidden = try zeroArray([1, 128])
            var cell = try zeroArray([1, 128])
            var context = [Float](repeating: 0, count: contextSize)
            var frames = [[String: Any]]()
            frames.reserveCapacity((audio.samples.count + chunkSize - 1) / chunkSize)
            let inferenceStart = ProcessInfo.processInfo.systemUptime
            for start in stride(from: 0, to: audio.samples.count, by: chunkSize) {
                try autoreleasepool {
                let end = min(start + chunkSize, audio.samples.count)
                for index in 0..<contextSize { inputPointer[index] = context[index] }
                for index in 0..<chunkSize {
                    inputPointer[contextSize + index] = start + index < end ? audio.samples[start + index] : 0
                }
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "audio_input": MLFeatureValue(multiArray: inputArray),
                    "hidden_state": MLFeatureValue(multiArray: hidden),
                    "cell_state": MLFeatureValue(multiArray: cell),
                ])
                let prediction = try model.prediction(from: provider)
                let probabilityArray = try checkedOutput(prediction, name: "vad_output", shape: [1, 1, 1])
                let probability = probabilityArray[0].floatValue
                guard probability.isFinite, (0...1).contains(probability) else {
                    throw CLIError(message: "Invalid VAD probability at source sample \(start)")
                }
                hidden = try checkedOutput(prediction, name: "new_hidden_state", shape: [1, 128])
                cell = try checkedOutput(prediction, name: "new_cell_state", shape: [1, 128])
                for index in 0..<contextSize { context[index] = inputPointer[chunkSize + index] }
                frames.append(["start_sample": start, "end_sample": end, "probability": probability])
                }
            }
            let inferenceSeconds = ProcessInfo.processInfo.systemUptime - inferenceStart
            stage = "export"
            try writeJSON([
                "file": input.path, "sha256": audio.sha256,
                "sample_count": audio.samples.count, "sample_rate": sampleRate,
                "duration": Double(audio.samples.count) / Double(sampleRate),
                "frames": frames,
                "model_load_seconds": loadSeconds,
                "inference_seconds": inferenceSeconds,
                "process_seconds_before_frame_export": ProcessInfo.processInfo.systemUptime - started,
                "final_frame_zero_padding_samples": paddingCount,
                "segmentation_performed": false,
            ], to: output.appendingPathComponent("frames.json"))
            try writeJSON([
                "status": "complete", "frames": frames.count,
                "wall_seconds": ProcessInfo.processInfo.systemUptime - started,
                "measurement_scope": "input read, model load, schema validation, inference and frame export; excludes final runtime metadata write",
            ], to: output.appendingPathComponent("runtime.json"))
            print("VAD: \(frames.count) continuous 32ms frames, \(audio.samples.count) source samples")
        } catch {
            try? writeJSON(["stage": stage, "error": error.localizedDescription],
                           to: output.appendingPathComponent("error.json"))
            throw error
        }
    }

    static func checkMetadata(_ value: Any?, expected: [String: [Int]]) throws {
        guard let rows = value as? [[String: Any]], rows.count == expected.count else {
            throw CLIError(message: "Model metadata feature count mismatch")
        }
        var seen = Set<String>()
        for row in rows {
            guard let name = row["name"] as? String, let shape = expected[name], seen.insert(name).inserted,
                  row["type"] as? String == "MultiArray", row["dataType"] as? String == "Float32",
                  row["isOptional"] as? String == "0", row["hasShapeFlexibility"] as? String == "0",
                  let encodedShape = row["shape"] as? String,
                  let actual = try JSONSerialization.jsonObject(with: Data(encodedShape.utf8)) as? [Int], actual == shape else {
                throw CLIError(message: "Model metadata does not match fixed 32ms Silero schema")
            }
        }
    }

    static func checkActual(_ features: [String: MLFeatureDescription], expected: [String: [Int]]) throws {
        guard Set(features.keys) == Set(expected.keys) else {
            throw CLIError(message: "Actual Core ML feature names do not match metadata")
        }
        for (name, shape) in expected {
            guard let feature = features[name], feature.type == .multiArray, !feature.isOptional,
                  let constraint = feature.multiArrayConstraint,
                  constraint.dataType == .float32, constraint.shape.map(\.intValue) == shape else {
                throw CLIError(message: "Actual Core ML feature \(name) does not match metadata shape/type")
            }
        }
    }

    static func describe(_ features: [String: MLFeatureDescription]) -> [String: Any] {
        Dictionary(uniqueKeysWithValues: features.map { name, feature in
            (name, ["type": String(describing: feature.type), "optional": feature.isOptional,
                    "data_type": feature.multiArrayConstraint.map { String(describing: $0.dataType) } ?? "none",
                    "shape": feature.multiArrayConstraint?.shape.map(\.intValue) ?? []] as [String: Any])
        })
    }

    static func zeroArray(_ shape: [Int]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        for index in 0..<result.count { result[index] = 0 }
        return result
    }

    static func checkedOutput(_ prediction: MLFeatureProvider, name: String, shape: [Int]) throws -> MLMultiArray {
        guard let result = prediction.featureValue(for: name)?.multiArrayValue,
              result.dataType == .float32, result.shape.map(\.intValue) == shape else {
            throw CLIError(message: "Unexpected prediction output \(name)")
        }
        guard (0..<result.count).allSatisfy({ result[$0].floatValue.isFinite }) else {
            throw CLIError(message: "Nonfinite prediction state \(name)")
        }
        return result
    }

    static func readInput(_ url: URL) throws -> (samples: [Float], sha256: String) {
        let data = try Data(contentsOf: url)
        func fail(_ message: String) -> CLIError { CLIError(message: "\(message): \(url.path)") }
        func u16(_ offset: Int) -> UInt16 {
            data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self)) }
        }
        func u32(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }
        func tag(_ offset: Int) -> String { String(decoding: data[offset..<offset + 4], as: UTF8.self) }
        guard data.count >= 12, tag(0) == "RIFF", tag(8) == "WAVE", Int(u32(4)) + 8 == data.count else {
            throw fail("Expected complete little-endian RIFF/WAVE file")
        }
        var offset = 12
        var validFormat = false
        var pcm: Range<Int>?
        while offset < data.count {
            guard offset + 8 <= data.count else { throw fail("Truncated WAV chunk header") }
            let kind = tag(offset), size = Int(u32(offset + 4))
            let start = offset + 8, end = start + size
            guard end <= data.count else { throw fail("Truncated WAV \(kind) chunk") }
            if kind == "fmt " {
                guard !validFormat, size >= 16, u16(start) == 3, u16(start + 2) == 1,
                      u32(start + 4) == UInt32(sampleRate), u32(start + 8) == UInt32(sampleRate * 4),
                      u16(start + 12) == 4, u16(start + 14) == 32 else {
                    throw fail("Expected one 16kHz mono float32 format-tag-3 chunk")
                }
                validFormat = true
            } else if kind == "data" {
                guard pcm == nil, size > 0, size % 4 == 0 else { throw fail("Invalid or repeated PCM data chunk") }
                pcm = start..<end
            }
            offset = end + size % 2
            guard offset <= data.count else { throw fail("Missing WAV chunk padding") }
        }
        guard validFormat, let pcm else { throw fail("Missing WAV format or PCM data") }
        let samples: [Float] = data.withUnsafeBytes { bytes in
            stride(from: pcm.lowerBound, to: pcm.upperBound, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
        guard samples.allSatisfy(\.isFinite) else { throw fail("Nonfinite PCM sample") }
        return (samples, sha256(data))
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func writeJSON(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    }
}
