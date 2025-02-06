flowchart TB
    subgraph Setup["Initial Setup"]
        A[setup_cameras.py] --> B[Camera Config]
        A --> C[NVR Settings]
        A --> D[Zone Config]
        B --> E[camera_config.json]
        C --> E
        D --> F[zone_config.json]
    end

    subgraph CameraSystem["Camera Management"]
        G[CameraManager] --> |Initialize| H[CameraStream]
        H --> |Frame Buffer| I[Frame Queue]
        H --> |Connection Management| J[RTSP Stream]
        G --> |Config Management| K[Camera Config]
    end

    subgraph FaceSystem["Face Recognition"]
        L[FaceRecognizer] --> |Detection| M[InsightFace]
        L --> |Embedding| N[FAISS Index]
        L --> |Storage| O[face_embeddings.pkl]
        L --> |Visitor DB| P[Visitors.csv]
    end

    subgraph Analytics["Traffic Analytics"]
        Q[TrafficAnalytics] --> |Position Tracking| R[Visitor Positions]
        Q --> |Zone Analysis| S[Zone Statistics]
        Q --> |Heat Map| T[Heat Map Generation]
        Q --> |Reports| U[Traffic Reports]
    end

    subgraph MainSystem["Store Monitoring System"]
        V[store_monitoring_system.py] --> |Initialize| W[System Components]
        W --> |Display| X[Multi-Camera Grid]
        W --> |Processing| Y[Frame Processing]
        W --> |Analytics| Z[Real-time Overlay]
    end

    %% Data Flow Connections
    E --> G
    F --> Q
    I --> Y
    Y --> L
    L --> Q
    Q --> Z
    Z --> X

    %% Storage Flow
    M --> O
    N --> P
    R --> U
    S --> U
    T --> |Save| AA[analytics/*.png]
    U --> |Save| AB[analytics/*.json]

    %% Real-time Display
    X --> |Show| AC[Display Grid]
    Z --> |Update| AC
    T --> |Overlay| AC

    style Setup fill:#f9f,stroke:#333,stroke-width:2px
    style CameraSystem fill:#bbf,stroke:#333,stroke-width:2px
    style FaceSystem fill:#bfb,stroke:#333,stroke-width:2px
    style Analytics fill:#fbf,stroke:#333,stroke-width:2px
    style MainSystem fill:#fbb,stroke:#333,stroke-width:2px
