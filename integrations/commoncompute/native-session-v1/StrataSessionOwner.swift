import Foundation
import StrataSession

public struct SessionDescriptor: Sendable {
    public let requestID: String
    public let modelDigest: Data
    public let reserveBytes: UInt64
    public let deadlineNanoseconds: UInt64
    public let promptTokens: UInt32
    public let outputTokens: UInt32
    public let contextTokens: UInt32
    public init(requestID: String, modelDigest: Data, reserveBytes: UInt64,
                deadlineNanoseconds: UInt64, promptTokens: UInt32,
                outputTokens: UInt32, contextTokens: UInt32) {
        self.requestID = requestID; self.modelDigest = modelDigest
        self.reserveBytes = reserveBytes; self.deadlineNanoseconds = deadlineNanoseconds
        self.promptTokens = promptTokens; self.outputTokens = outputTokens; self.contextTokens = contextTokens
    }
}
public struct SessionView: Sendable {
    public let requestID: String
    public let modelDigest: Data
    public let reservedBytes: UInt64
    public let revision: UInt64
    public let state: UInt32
    public let reason: UInt32
    public let promptTokens: UInt32
    public let completionTokens: UInt32
    public let runtimeQualified: Bool
}
public struct SessionTotals: Sendable {
    public let reservedBytes: UInt64
    public let active: UInt32
    public let queued: UInt32
    public let records: UInt32
    public let draining: Bool
}
public enum SessionError: Error { case invalidDescriptor, closed, native(UInt32) }

/// Trusted engine-side owner, after sealed input decryption/manifest validation.
/// No transport, decryption, plaintext inference, batching or model execution.
/// Executor must hold this owner, confirm GPU safe points, then call finish.
public actor StrataSessionOwner {
    private var handle: OpaquePointer?
    public init(memoryLimitBytes: UInt64, maximumQueued: UInt32 = 4, maximumRecords: UInt32 = 32) throws {
        var config = strata_session_config_v1()
        config.abi_version = 1; config.struct_bytes = UInt32(MemoryLayout<strata_session_config_v1>.size)
        config.memory_limit_bytes = memoryLimitBytes; config.max_queued = maximumQueued
        config.max_active = 1; config.max_records = maximumRecords
        var created: OpaquePointer?
        let result = strata_session_create_v1(&config, &created)
        guard result == STRATA_SESSION_OK else { throw SessionError.native(result.rawValue) }
        handle = created
    }
    isolated deinit {
        // Refuses to destroy active work. Explicit joined close is mandatory;
        // violating ownership can leak metadata, never force-free active work.
        var owned = handle
        if owned != nil { _ = strata_session_destroy_v1(&owned) }
    }
    private func pointer() throws -> OpaquePointer {
        guard let handle else { throw SessionError.closed }
        return handle
    }
    private func checked(_ result: strata_session_result) throws {
        guard result == STRATA_SESSION_OK else { throw SessionError.native(result.rawValue) }
    }
    private func identifier(_ id: String) throws -> String {
        let bytes = Array(id.utf8)
        func alphanumeric(_ byte: UInt8) -> Bool {
            (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122) || (byte >= 48 && byte <= 57)
        }
        guard let first = bytes.first, bytes.count < 128, alphanumeric(first),
              bytes.allSatisfy({ alphanumeric($0) || [UInt8(46), 95, 58, 45].contains($0) }) else {
            throw SessionError.invalidDescriptor
        }
        return id
    }
    private func view(_ value: strata_session_snapshot_v1) -> SessionView {
        var value = value
        let id = withUnsafeBytes(of: &value.request_id) { buffer in
            String(decoding: buffer.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        let digest = withUnsafeBytes(of: &value.model_digest) { Data($0) }
        return SessionView(requestID: id, modelDigest: digest, reservedBytes: value.reserved_bytes,
                           revision: value.revision, state: value.state, reason: value.reason,
                           promptTokens: value.prompt_tokens, completionTokens: value.completion_tokens,
                           runtimeQualified: value.native_runtime_qualified != 0)
    }
    public func submit(_ descriptor: SessionDescriptor, nowNanoseconds: UInt64) throws {
        let bytes = Array(try identifier(descriptor.requestID).utf8)
        guard !bytes.isEmpty, bytes.count < 128, !bytes.contains(0), descriptor.modelDigest.count == 32 else {
            throw SessionError.invalidDescriptor
        }
        var request = strata_session_request_v1()
        request.abi_version = 1; request.struct_bytes = UInt32(MemoryLayout<strata_session_request_v1>.size)
        withUnsafeMutableBytes(of: &request.request_id) { $0.copyBytes(from: bytes) }
        withUnsafeMutableBytes(of: &request.model_digest) { $0.copyBytes(from: descriptor.modelDigest) }
        request.reserve_bytes = descriptor.reserveBytes; request.deadline_monotonic_ns = descriptor.deadlineNanoseconds
        request.prompt_tokens = descriptor.promptTokens; request.max_output_tokens = descriptor.outputTokens
        request.context_tokens = descriptor.contextTokens
        try checked(strata_session_submit_v1(try pointer(), &request, nowNanoseconds))
    }
    public func next(nowNanoseconds: UInt64) throws -> SessionView {
        var result = strata_session_snapshot_v1()
        try checked(strata_session_next_v1(try pointer(), nowNanoseconds, &result))
        return view(result)
    }
    public func read(_ id: String) throws -> SessionView {
        var result = strata_session_snapshot_v1()
        try checked(strata_session_read_v1(try pointer(), try identifier(id), &result))
        return view(result)
    }
    public func addTokens(_ id: String, delta: UInt32) throws {
        try checked(strata_session_add_tokens_v1(try pointer(), try identifier(id), delta))
    }
    public func cancel(_ id: String) throws { try checked(strata_session_cancel_v1(try pointer(), try identifier(id))) }
    public func check(_ id: String, nowNanoseconds: UInt64) throws {
        try checked(strata_session_check_v1(try pointer(), try identifier(id), nowNanoseconds))
    }
    public func finish(_ id: String, state: UInt32, reason: UInt32) throws {
        try checked(strata_session_finish_v1(try pointer(), try identifier(id), state, reason))
    }
    public func forget(_ id: String) throws { try checked(strata_session_forget_v1(try pointer(), try identifier(id))) }
    public func drain() throws { try checked(strata_session_drain_v1(try pointer())) }
    public func totals() throws -> SessionTotals {
        var result = strata_session_totals_v1()
        try checked(strata_session_totals_v1_read(try pointer(), &result))
        return SessionTotals(reservedBytes: result.reserved_bytes, active: result.active,
                             queued: result.queued, records: result.records, draining: result.draining != 0)
    }
    public func close() throws {
        if handle == nil { return }
        try checked(strata_session_destroy_v1(&handle))
    }
}
