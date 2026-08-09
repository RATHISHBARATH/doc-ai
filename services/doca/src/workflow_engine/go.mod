module github.com/doc-ai/doca/workflow-engine

go 1.22

require (
    github.com/gin-gonic/gin v1.9.1           // for health/metrics endpoints
    github.com/segmentio/kafka-go v0.4.47    // Kafka client
    github.com/nats-io/nats.go v1.34.0       // NATS client
    github.com/hashicorp/go-uuid v1.0.3      // for task ID generation
    github.com/prometheus/client_golang v1.19.0 // metrics
)