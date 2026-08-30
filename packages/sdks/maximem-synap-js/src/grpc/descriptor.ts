/**
 * GENERATED FILE. Do not edit by hand.
 *
 * Source:    synap/proto/synap_service.proto
 * Generator: synap/sdk/scripts/gen_proto_descriptor.mjs
 * Proto SHA: 342112aaf000c62b
 *
 * The proto is inlined as a JSON descriptor rather than loaded from disk so
 * that bundlers and Vercel file tracing cannot lose it. Regenerate with
 * `npm run sync-behavior` in synap/sdk/js.
 */

/** sha256 (first 16 hex chars) of the proto this descriptor was built from. */
export const PROTO_SHA = '342112aaf000c62b';

export const SYNAP_PROTO_DESCRIPTOR = {
  "nested": {
    "synap": {
      "nested": {
        "v1": {
          "nested": {
            "SynapService": {
              "methods": {
                "Listen": {
                  "requestType": "StreamEvent",
                  "requestStream": true,
                  "responseType": "StreamResponse",
                  "responseStream": true
                },
                "IngestTelemetry": {
                  "requestType": "TelemetryEvent",
                  "requestStream": true,
                  "responseType": "TelemetryAck"
                }
              }
            },
            "StreamEvent": {
              "oneofs": {
                "payload": {
                  "oneof": [
                    "conversation_event",
                    "heartbeat_ping",
                    "session_control",
                    "context_used",
                    "context_assembled"
                  ]
                }
              },
              "fields": {
                "conversation_event": {
                  "type": "ConversationEvent",
                  "id": 1
                },
                "heartbeat_ping": {
                  "type": "HeartbeatPing",
                  "id": 2
                },
                "session_control": {
                  "type": "SessionControl",
                  "id": 3
                },
                "context_used": {
                  "type": "ContextUsedEvent",
                  "id": 4
                },
                "context_assembled": {
                  "type": "ContextAssembledEvent",
                  "id": 5
                }
              }
            },
            "ContextUsedEvent": {
              "fields": {
                "bundle_id": {
                  "type": "string",
                  "id": 1
                },
                "conversation_id": {
                  "type": "string",
                  "id": 2
                },
                "user_id": {
                  "type": "string",
                  "id": 3
                },
                "customer_id": {
                  "type": "string",
                  "id": 4
                },
                "served_item_ids": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 5
                },
                "timestamp_ms": {
                  "type": "int64",
                  "id": 6
                },
                "scope": {
                  "type": "string",
                  "id": 7
                },
                "source_bundle_ids": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 8
                }
              }
            },
            "ContextAssembledEvent": {
              "fields": {
                "correlation_id": {
                  "type": "string",
                  "id": 1
                },
                "conversation_id": {
                  "type": "string",
                  "id": 2
                },
                "user_id": {
                  "type": "string",
                  "id": 3
                },
                "customer_id": {
                  "type": "string",
                  "id": 4
                },
                "final_item_ids": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 5
                },
                "final_total_tokens": {
                  "type": "int32",
                  "id": 6
                },
                "compaction_id": {
                  "type": "string",
                  "id": 7
                },
                "recent_turn_count": {
                  "type": "int32",
                  "id": 8
                },
                "compaction_end_timestamp": {
                  "type": "string",
                  "id": 9
                },
                "assembly_source": {
                  "type": "string",
                  "id": 10
                },
                "assembly_duration_ms": {
                  "type": "int32",
                  "id": 11
                },
                "cache_hit": {
                  "type": "bool",
                  "id": 12
                },
                "timestamp_ms": {
                  "type": "int64",
                  "id": 13
                },
                "sdk_version": {
                  "type": "string",
                  "id": 14
                }
              }
            },
            "ConversationEvent": {
              "fields": {
                "event_type": {
                  "type": "string",
                  "id": 1
                },
                "conversation_id": {
                  "type": "string",
                  "id": 2
                },
                "user_id": {
                  "type": "string",
                  "id": 3
                },
                "role": {
                  "type": "string",
                  "id": 4
                },
                "content": {
                  "type": "string",
                  "id": 5
                },
                "customer_id": {
                  "type": "string",
                  "id": 6
                },
                "session_id": {
                  "type": "string",
                  "id": 7
                },
                "metadata": {
                  "keyType": "string",
                  "type": "string",
                  "id": 8
                },
                "timestamp_ms": {
                  "type": "int64",
                  "id": 9
                },
                "tool_name": {
                  "type": "string",
                  "id": 10
                },
                "tool_args_json": {
                  "type": "string",
                  "id": 11
                },
                "search_queries": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 12
                },
                "context_types": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 13
                }
              }
            },
            "HeartbeatPing": {
              "fields": {
                "timestamp_ms": {
                  "type": "int64",
                  "id": 1
                }
              }
            },
            "SessionControl": {
              "fields": {
                "action": {
                  "type": "string",
                  "id": 1
                },
                "session_id": {
                  "type": "string",
                  "id": 2
                },
                "conversation_id": {
                  "type": "string",
                  "id": 3
                },
                "user_id": {
                  "type": "string",
                  "id": 4
                },
                "customer_id": {
                  "type": "string",
                  "id": 5
                }
              }
            },
            "StreamResponse": {
              "oneofs": {
                "payload": {
                  "oneof": [
                    "context_bundle",
                    "heartbeat_pong",
                    "signal"
                  ]
                }
              },
              "fields": {
                "context_bundle": {
                  "type": "ContextBundleProto",
                  "id": 1
                },
                "heartbeat_pong": {
                  "type": "HeartbeatPong",
                  "id": 2
                },
                "signal": {
                  "type": "StreamSignal",
                  "id": 3
                }
              }
            },
            "HeartbeatPong": {
              "fields": {
                "timestamp_ms": {
                  "type": "int64",
                  "id": 1
                }
              }
            },
            "StreamSignal": {
              "fields": {
                "signal_type": {
                  "type": "string",
                  "id": 1
                },
                "reason": {
                  "type": "string",
                  "id": 2
                },
                "metadata": {
                  "keyType": "string",
                  "type": "string",
                  "id": 3
                }
              }
            },
            "ContextBundleProto": {
              "fields": {
                "bundle_id": {
                  "type": "string",
                  "id": 1
                },
                "decision_id": {
                  "type": "string",
                  "id": 2
                },
                "items_by_type": {
                  "keyType": "string",
                  "type": "ContextItemList",
                  "id": 3
                },
                "total_tokens": {
                  "type": "int32",
                  "id": 4
                },
                "token_budget": {
                  "type": "int32",
                  "id": 5
                },
                "budget_exceeded": {
                  "type": "bool",
                  "id": 6
                },
                "retrieval_mode": {
                  "type": "string",
                  "id": 7
                },
                "sources_queried": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 8
                },
                "degradation_level": {
                  "type": "string",
                  "id": 9
                },
                "warnings": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 10
                },
                "created_at": {
                  "type": "string",
                  "id": 11
                },
                "retrieval_time_ms": {
                  "type": "int32",
                  "id": 12
                },
                "cache_hit": {
                  "type": "bool",
                  "id": 13
                },
                "search_queries": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 14
                },
                "anticipation_user_id": {
                  "type": "string",
                  "id": 15
                },
                "anticipation_customer_id": {
                  "type": "string",
                  "id": 16
                },
                "anticipation_conversation_id": {
                  "type": "string",
                  "id": 17
                },
                "search_keywords": {
                  "rule": "repeated",
                  "type": "string",
                  "id": 18
                },
                "bundle_type": {
                  "type": "string",
                  "id": 19
                },
                "conversation_context": {
                  "type": "ConversationContextProto",
                  "id": 20
                },
                "bundle_confidence": {
                  "type": "float",
                  "id": 21
                },
                "origin_pattern_id": {
                  "type": "string",
                  "id": 22
                },
                "ttl_hint_seconds": {
                  "type": "int32",
                  "id": 23
                },
                "anticipation_scope_rung": {
                  "type": "string",
                  "id": 24
                }
              }
            },
            "ConversationContextProto": {
              "fields": {
                "summary": {
                  "type": "string",
                  "id": 1
                },
                "current_state_json": {
                  "type": "string",
                  "id": 2
                },
                "key_extractions_json": {
                  "type": "string",
                  "id": 3
                },
                "recent_turns": {
                  "rule": "repeated",
                  "type": "RecentTurnProto",
                  "id": 4
                },
                "compaction_id": {
                  "type": "string",
                  "id": 5
                },
                "compacted_at": {
                  "type": "string",
                  "id": 6
                },
                "conversation_id": {
                  "type": "string",
                  "id": 7
                }
              }
            },
            "RecentTurnProto": {
              "fields": {
                "role": {
                  "type": "string",
                  "id": 1
                },
                "content": {
                  "type": "string",
                  "id": 2
                },
                "timestamp": {
                  "type": "string",
                  "id": 3
                }
              }
            },
            "ContextItemList": {
              "fields": {
                "items": {
                  "rule": "repeated",
                  "type": "ContextItemProto",
                  "id": 1
                }
              }
            },
            "ContextItemProto": {
              "fields": {
                "item_id": {
                  "type": "string",
                  "id": 1
                },
                "content": {
                  "type": "string",
                  "id": 2
                },
                "context_type": {
                  "type": "string",
                  "id": 3
                },
                "source": {
                  "type": "string",
                  "id": 4
                },
                "similarity_score": {
                  "type": "float",
                  "id": 5
                },
                "relevance_score": {
                  "type": "float",
                  "id": 6
                },
                "confidence": {
                  "type": "float",
                  "id": 7
                },
                "scope": {
                  "type": "string",
                  "id": 8
                },
                "entity_id": {
                  "type": "string",
                  "id": 9
                },
                "created_at": {
                  "type": "string",
                  "id": 10
                },
                "event_date": {
                  "type": "string",
                  "id": 11
                },
                "valid_until": {
                  "type": "string",
                  "id": 12
                },
                "temporal_category": {
                  "type": "string",
                  "id": 13
                },
                "temporal_confidence": {
                  "type": "float",
                  "id": 14
                }
              }
            },
            "TelemetryEvent": {
              "fields": {
                "event_type": {
                  "type": "string",
                  "id": 1
                },
                "instance_id": {
                  "type": "string",
                  "id": 2
                },
                "client_id": {
                  "type": "string",
                  "id": 3
                },
                "correlation_id": {
                  "type": "string",
                  "id": 4
                },
                "timestamp_ms": {
                  "type": "int64",
                  "id": 5
                },
                "latency_ms": {
                  "type": "int32",
                  "id": 6
                },
                "status": {
                  "type": "string",
                  "id": 7
                },
                "error_code": {
                  "type": "string",
                  "id": 8
                },
                "scope": {
                  "type": "string",
                  "id": 9
                },
                "cache_status": {
                  "type": "string",
                  "id": 10
                },
                "attempt": {
                  "type": "int32",
                  "id": 11
                },
                "http_method": {
                  "type": "string",
                  "id": 12
                },
                "http_path": {
                  "type": "string",
                  "id": 13
                },
                "http_status_code": {
                  "type": "int32",
                  "id": 14
                },
                "metadata": {
                  "keyType": "string",
                  "type": "string",
                  "id": 15
                },
                "sdk_version": {
                  "type": "string",
                  "id": 16
                },
                "batch_id": {
                  "type": "string",
                  "id": 17
                }
              }
            },
            "TelemetryAck": {
              "fields": {
                "status": {
                  "type": "string",
                  "id": 1
                },
                "events_received": {
                  "type": "int32",
                  "id": 2
                }
              }
            }
          }
        }
      }
    }
  }
} as const;
