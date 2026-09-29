import Foundation
import AVFoundation
@main struct Normalize {
 static func main() throws {
  let a = CommandLine.arguments
  guard a.count == 3 else { fatalError("normalize input output.wav") }
  let out = URL(fileURLWithPath:a[2])
  guard !FileManager.default.fileExists(atPath:out.path) else { fatalError("Output exists") }
  let input = try AVAudioFile(forReading: URL(fileURLWithPath:a[1]))
  let format = AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:16000,channels:1,interleaved:false)!
  print("SOURCE \(input.processingFormat)"); fflush(stdout)
  let converter = AVAudioConverter(from:input.processingFormat,to:format)!
  converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
  print("CONVERTER ready"); fflush(stdout)
  let output = try AVAudioFile(forWriting:out,settings:format.settings)
  print("OUTPUT ready"); fflush(stdout)
  let ib = AVAudioPCMBuffer(pcmFormat:input.processingFormat,frameCapacity:65536)!
  let ob = AVAudioPCMBuffer(pcmFormat:format,frameCapacity:32768)!
  var ended = false
  while !ended {
   var error:NSError?
   var readError:Error?
   let status = converter.convert(to:ob,error:&error) { needed, status in
    if input.framePosition >= input.length { status.pointee = .endOfStream; return nil }
    do { try input.read(into:ib,frameCount:min(needed, ib.frameCapacity, AVAudioFrameCount(input.length - input.framePosition))) }
    catch { readError = error; status.pointee = .endOfStream; return nil }
    status.pointee = ib.frameLength == 0 ? .endOfStream : .haveData
    return ib.frameLength == 0 ? nil : ib
   }
   if let error = readError { throw error }
   if let error { throw error }
   if ob.frameLength > 0 { try output.write(from:ob) }
   ended = status == .endOfStream
  }
  print("\(input.length) source frames @ \(input.processingFormat.sampleRate); \(output.length) output frames @ 16000; mono float32; no trimming")
 }
}
