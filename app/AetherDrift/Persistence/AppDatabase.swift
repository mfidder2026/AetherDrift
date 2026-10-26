import Foundation
import GRDB

struct Conversation: Codable, Sendable, Identifiable, Hashable, FetchableRecord, MutablePersistableRecord {
    var id: Int64?
    var title: String
    var createdAt: Date
    var updatedAt: Date

    mutating func didInsert(_ inserted: InsertionSuccess) { id = inserted.rowID }
}

/// Per-answer details shown in the footer and kept with the message.
struct MessageMeta: Codable, Sendable, Equatable {
    var decodeTps: Double?
    var completionTokens: Int?
    var mode: String?
    var model: String?
    var finishReason: String?
    var errorCode: String?
}

struct StoredMessage: Codable, Sendable, Identifiable, Hashable, FetchableRecord, MutablePersistableRecord {
    static let databaseTableName = "message"
    var id: Int64?
    var conversationId: Int64
    var role: String  // user | assistant
    var content: String
    var createdAt: Date
    var meta: String?  // JSON-encoded MessageMeta

    mutating func didInsert(_ inserted: InsertionSuccess) { id = inserted.rowID }

    var decodedMeta: MessageMeta? {
        meta.flatMap { try? JSONDecoder().decode(MessageMeta.self, from: Data($0.utf8)) }
    }
}

/// SQLite store for conversations (start-prompt-v2 §8.3).
final class AppDatabase: Sendable {
    let writer: any DatabaseWriter

    init(_ writer: any DatabaseWriter) throws {
        self.writer = writer
        try Self.migrator.migrate(writer)
    }

    static func onDisk() throws -> AppDatabase {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "AetherDrift")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return try AppDatabase(DatabaseQueue(path: dir.appending(path: "history.sqlite").path))
    }

    static func inMemory() throws -> AppDatabase {
        try AppDatabase(DatabaseQueue())
    }

    private static var migrator: DatabaseMigrator {
        var m = DatabaseMigrator()
        m.registerMigration("v1") { db in
            try db.create(table: "conversation") { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("title", .text).notNull()
                t.column("createdAt", .datetime).notNull()
                t.column("updatedAt", .datetime).notNull().indexed()
            }
            try db.create(table: "message") { t in
                t.autoIncrementedPrimaryKey("id")
                t.belongsTo("conversation", onDelete: .cascade).notNull()
                t.column("role", .text).notNull()
                t.column("content", .text).notNull()
                t.column("createdAt", .datetime).notNull()
                t.column("meta", .text)
            }
        }
        return m
    }

    // MARK: Queries

    func conversations() throws -> [Conversation] {
        try writer.read { db in
            try Conversation.order(Column("updatedAt").desc).fetchAll(db)
        }
    }

    func messages(in conversationID: Int64) throws -> [StoredMessage] {
        try writer.read { db in
            try StoredMessage
                .filter(Column("conversationId") == conversationID)
                .order(Column("id"))
                .fetchAll(db)
        }
    }

    func createConversation(title: String) throws -> Conversation {
        try writer.write { db in
            var c = Conversation(id: nil, title: title, createdAt: .now, updatedAt: .now)
            try c.insert(db)
            return c
        }
    }

    func rename(_ id: Int64, to title: String) throws {
        try writer.write { db in
            _ = try Conversation.filter(key: id).updateAll(db, Column("title").set(to: title))
        }
    }

    func deleteConversation(_ id: Int64) throws {
        try writer.write { db in _ = try Conversation.deleteOne(db, key: id) }
    }

    @discardableResult
    func save(_ message: StoredMessage) throws -> StoredMessage {
        try writer.write { db in
            var m = message
            try m.save(db)
            _ = try Conversation.filter(key: m.conversationId)
                .updateAll(db, Column("updatedAt").set(to: Date.now))
            return m
        }
    }

    func deleteMessages(in conversationID: Int64, from messageID: Int64) throws {
        try writer.write { db in
            _ = try StoredMessage
                .filter(Column("conversationId") == conversationID && Column("id") >= messageID)
                .deleteAll(db)
        }
    }
}
