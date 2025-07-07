-- Shopping Assistant Database Initialization
-- This file is used by docker-compose to initialize the PostgreSQL database

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create additional schemas if needed
-- CREATE SCHEMA IF NOT EXISTS analytics;
-- CREATE SCHEMA IF NOT EXISTS reporting;

-- Create indexes for commonly queried fields (will be created by Alembic migrations)
-- These are just examples of what might be useful

-- Performance optimizations
-- Set some basic PostgreSQL configurations for better performance
-- (In production, these should be set in postgresql.conf)

-- Grant permissions to the application user
GRANT ALL PRIVILEGES ON DATABASE shopping_assistant TO shopping_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO shopping_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO shopping_user;

-- Set default permissions for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO shopping_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO shopping_user;

-- Optional: Create a read-only user for analytics/reporting
-- CREATE USER analytics_user WITH PASSWORD 'analytics_password';
-- GRANT CONNECT ON DATABASE shopping_assistant TO analytics_user;
-- GRANT USAGE ON SCHEMA public TO analytics_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_user;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO analytics_user; 