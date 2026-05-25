-- Run this ONCE in a Databricks SQL Editor query (NOT inside the DLT pipeline).
-- Grants the running user write privileges on all 4 schemas of the advdatafinal catalog.

GRANT USE CATALOG ON CATALOG advdatafinal TO `lfrenteria33@gmail.com`;

GRANT USE SCHEMA, CREATE TABLE, MODIFY, CREATE MATERIALIZED VIEW, CREATE VOLUME
    ON SCHEMA advdatafinal.raw          TO `lfrenteria33@gmail.com`;
GRANT USE SCHEMA, CREATE TABLE, MODIFY, CREATE MATERIALIZED VIEW
    ON SCHEMA advdatafinal.datos_masked TO `lfrenteria33@gmail.com`;
GRANT USE SCHEMA, CREATE TABLE, MODIFY, CREATE MATERIALIZED VIEW
    ON SCHEMA advdatafinal.silver       TO `lfrenteria33@gmail.com`;
GRANT USE SCHEMA, CREATE TABLE, MODIFY, CREATE MATERIALIZED VIEW
    ON SCHEMA advdatafinal.gold         TO `lfrenteria33@gmail.com`;

GRANT READ VOLUME ON VOLUME advdatafinal.raw.landing TO `lfrenteria33@gmail.com`;
