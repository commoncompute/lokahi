import Foundation

@main struct VerifySession {
    static func main() async throws {
        let owner = try StrataSessionOwner(memoryLimitBytes: 1024)
        let descriptor = SessionDescriptor(requestID: "fixture", modelDigest: Data(repeating: 7, count: 32),
                                           reserveBytes: 512, deadlineNanoseconds: 1000,
                                           promptTokens: 64, outputTokens: 32, contextTokens: 128)
        try await owner.submit(descriptor, nowNanoseconds: 1)
        let first = try await owner.next(nowNanoseconds: 2)
        precondition(first.state == 2 && first.requestID == "fixture" && !first.runtimeQualified)
        do { _ = try await owner.read("fixture\0suffix"); fatalError("NUL truncated") } catch SessionError.invalidDescriptor {}
        do { try await owner.cancel("fixture\0suffix"); fatalError("NUL truncated") } catch SessionError.invalidDescriptor {}
        do { _ = try await owner.read("é"); fatalError("non-ASCII identity") } catch SessionError.invalidDescriptor {}
        try await owner.addTokens("fixture", delta: 8)
        try await owner.cancel("fixture")
        let cancelling = try await owner.totals()
        precondition(cancelling.reservedBytes == 512 && cancelling.active == 1)
        do { try await owner.close(); fatalError("active work destroyed") } catch SessionError.native(5) {}
        try await owner.finish("fixture", state: 6, reason: 4)
        let final = try await owner.read("fixture")
        let totals = try await owner.totals()
        precondition(final.completionTokens == 8 && final.state == 6 && final.revision > first.revision)
        precondition(totals.reservedBytes == 0 && totals.active == 0)
        do { try await owner.finish("fixture", state: 6, reason: 4); fatalError("duplicate terminal") } catch SessionError.native(6) {}
        try await owner.close()
        print("Swift/native lifecycle parity passed; synthetic descriptor, no model execution")
    }
}
