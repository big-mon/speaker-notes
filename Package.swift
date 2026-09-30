// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SpeakerNotes",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "normalize", targets: ["Normalize"]),
        .executable(name: "apple-transcribe", targets: ["AppleTranscribe"]),
        .executable(name: "silero-vad-frames", targets: ["SileroVADFrames"]),
        .executable(name: "fluid-diarize", targets: ["FluidDiarize"]),
    ],
    dependencies: [.package(path: "vendor/FluidAudio", traits: [])],
    targets: [
        .executableTarget(name: "Normalize"),
        .executableTarget(name: "AppleTranscribe"),
        .executableTarget(name: "SileroVADFrames"),
        .executableTarget(name: "FluidDiarize", dependencies: [
            .product(name: "FluidAudio", package: "FluidAudio"),
        ]),
    ]
)
