// Rasterize SVG files to square PNGs with the system SVG renderer.
// Usage: svg2png <size> <in.svg> <out.png> [<in.svg> <out.png> ...]
import AppKit
let args = CommandLine.arguments
let size = Int(args[1])!
var i = 2
while i + 1 < args.count {
    guard let image = NSImage(contentsOf: URL(fileURLWithPath: args[i])) else {
        FileHandle.standardError.write("cannot load \(args[i])\n".data(using: .utf8)!); exit(1)
    }
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = NSSize(width: size, height: size)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    NSGraphicsContext.current?.imageInterpolation = .high
    image.draw(in: NSRect(x: 0, y: 0, width: size, height: size))
    NSGraphicsContext.restoreGraphicsState()
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: args[i + 1]))
    i += 2
}
