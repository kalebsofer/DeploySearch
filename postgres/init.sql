CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    query TEXT NOT NULL,
    selected_document TEXT NOT NULL,
    similarity_score FLOAT NOT NULL,
    feedback_received BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_session ON search_logs(user_id, session_id);
CREATE INDEX idx_created_at ON search_logs(created_at);