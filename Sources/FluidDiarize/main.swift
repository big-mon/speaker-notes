import Foundation
import CoreML
import FluidAudio

@main struct CLI {
    static func main() async throws {
        ModelHub.offlineMode = true
        let args = CommandLine.arguments
        guard args.count == 4 || args.count == 5 else { fatalError("fluid-diarize audio.wav model-directory output-directory") }
        let output = URL(fileURLWithPath: args[3])
        guard !FileManager.default.fileExists(atPath: output.path) else { fatalError("Output exists") }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let started = Date()
        var config = OfflineDiarizerConfig()
        config.segmentation.stepRatio = 0.1
        config.postProcessing.exclusiveSegments = false
        config.embedding.minSegmentDurationSeconds = 0.5
        let manager = OfflineDiarizerManager(config: config)
        let ml = MLModelConfiguration()
        ml.computeUnits = .all
        try await manager.prepareModels(directory: URL(fileURLWithPath: args[2]), configuration: ml)
        let loaded = Date()
        let passes = args.count == 5 ? Int(args[4])! : 2
        for pass in 1...passes {
            let begin = Date()
            let result = try await manager.process(URL(fileURLWithPath: args[1])) { done, total in
                if done % 100 == 0 || done == total { print("Segmentation \(done)/\(total)"); fflush(stdout) }
            }
            let ended = Date()
            let segments: [[String: Any]] = result.segments.map { ["start": $0.startTimeSeconds, "end": $0.endTimeSeconds, "speaker": $0.speakerId] }
            let payload: [String: Any] = ["engine": "FluidAudio Community-1 offline VBx", "pass": pass,
                "load_seconds": loaded.timeIntervalSince(started), "processing_seconds": ended.timeIntervalSince(begin),
                "elapsed_since_launch": ended.timeIntervalSince(started), "segments": segments,
                "configuration": String(reflecting: config), "compute_units": "all; FBank cpuOnly"]
            try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted,.sortedKeys]).write(to: output.appendingPathComponent("pass-\(pass).json"))
            print("PASS \(pass) DONE \(ended.timeIntervalSince(begin)) seconds"); fflush(stdout)
        }
    }
}
