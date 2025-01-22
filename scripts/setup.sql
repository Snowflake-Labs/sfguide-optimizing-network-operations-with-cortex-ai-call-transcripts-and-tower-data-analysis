ALTER SESSION SET query_tag = '{"origin":"sf_sit-is","name":"telco_opt_nw_ops","version":{"major":1, "minor":0},"attributes":{"is_quickstart":0, "source":"sql"}}';

-- Switch to ACCOUNTADMIN role
USE ROLE ACCOUNTADMIN;

-- Create a new role for data scientists
CREATE OR REPLACE ROLE TELCO_NETWORK_OPTIMIZATION_ANALYST;

-- set my_user_var variable to equal the logged-in user
SET my_user_var = (SELECT  '"' || CURRENT_USER() || '"' );

-- Grant role to current user
GRANT ROLE TELCO_NETWORK_OPTIMIZATION_ANALYST TO USER identifier($my_user_var);

-- Create a warehouse with specified configuration
CREATE OR REPLACE WAREHOUSE TELCO_NETWORK_OPTIMIZATION_ANALYST_WH 
    SCALING_POLICY = 'STANDARD', 
    WAREHOUSE_SIZE = 'XSMALL', 
    WAREHOUSE_TYPE = 'STANDARD', 
    AUTO_RESUME = true, 
    AUTO_SUSPEND = 60, 
    MAX_CONCURRENCY_LEVEL = 8, 
    STATEMENT_TIMEOUT_IN_SECONDS = 172800;

-- Create a database
CREATE OR REPLACE DATABASE TELCO_NETWORK_OPTIMIZATION_PROD;

-- Create schema within the database
CREATE SCHEMA TELCO_NETWORK_OPTIMIZATION_PROD.RAW;

-- Create or replace a CSV file format in the RAW_POS schema
CREATE OR REPLACE FILE FORMAT TELCO_NETWORK_OPTIMIZATION_PROD.RAW.CSV_TELCO_NW_OPT
    TYPE = 'csv'
    NULL_IF = ('NULL', 'null', '', '\N', '\\N')
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1; 

-- Create or replace an S3 stage in the RAW schema
CREATE OR REPLACE STAGE TELCO_NETWORK_OPTIMIZATION_PROD.RAW.S3LOAD_TELCO_NW_OPT
    COMMENT = 'Quickstarts S3 Stage Connection',
    URL = 's3://sfquickstarts/sfguide_optimizing_network_operations_with_cortex_ai_call_transcripts_and_tower_data_analysis/',
    FILE_FORMAT = TELCO_NETWORK_OPTIMIZATION_PROD.RAW.CSV_TELCO_NW_OPT;

    
-- Create or replace the CELL_TOWER table in the RAW schema
create or replace TABLE TELCO_NETWORK_OPTIMIZATION_PROD.RAW.CELL_TOWER (
	CELL_ID NUMBER(38,0),
	CALL_RELEASE_CODE NUMBER(38,0),
	LOOKUP_ID NUMBER(38,0),
	HOME_NETWORK_TAP_CODE VARCHAR(16777216),
	SERVING_NETWORK_TAP_CODE VARCHAR(16777216),
	IMSI_PREFIX NUMBER(38,0),
	IMEI_PREFIX NUMBER(38,0),
	HOME_NETWORK_NAME VARCHAR(16777216),
	HOME_NETWORK_COUNTRY VARCHAR(16777216),
	BID_SERVING_NETWORK NUMBER(38,0),
	BID_DESCRIPTION VARCHAR(16777216),
	SERVICE_CATEGORY VARCHAR(16777216),
	CALL_EVENT_DESCRIPTION VARCHAR(16777216),
	ORIG_ID NUMBER(38,0),
	EVENT_DATE DATE,
	IMSI_SUFFIX NUMBER(38,0),
	IMEI_SUFFIX NUMBER(38,0),
	LOCATION_AREA_CODE NUMBER(38,0),
	CHARGED_UNITS NUMBER(38,0),
	MSISDN NUMBER(38,0),
	EVENT_DTTM TIMESTAMP_NTZ(9),
	CALL_ID VARCHAR(16777216),
	CAUSE_CODE_SHORT_DESCRIPTION VARCHAR(16777216),
	CAUSE_CODE_LONG_DESCRIPTION VARCHAR(16777216),
	CELL_LATITUDE NUMBER(38,9),
	CELL_LONGITUDE NUMBER(38,9),
	SENDER_NAME VARCHAR(16777216),
	VENDOR_NAME VARCHAR(16777216),
	HOSTNAME VARCHAR(16777216),
	TIMESTAMP TIMESTAMP_NTZ(9),
	DURATION NUMBER(38,0),
	MANAGED_ELEMENT NUMBER(38,0),
	ENODEB_FUNCTION NUMBER(38,0),
	WINDOW_START_AT TIMESTAMP_NTZ(9),
	WINDOW_END_AT TIMESTAMP_NTZ(9),
	INDEX VARCHAR(16777216),
	UE_MEAS_CONTROL NUMBER(38,0),
	PM_UE_MEAS_CONTROL NUMBER(38,0),
	PM_ACTIVE_UE_DL_MAX NUMBER(38,2),
	PM_ACTIVE_UE_DL_SUM NUMBER(38,2),
	PM_ACTIVE_UE_UL_MAX NUMBER(38,2),
	PM_ACTIVE_UE_UL_SUM NUMBER(38,2),
	PM_RRC_CONN_MAX NUMBER(38,2),
	PM_PDCP_LAT_TIME_DL NUMBER(38,2),
	PM_PDCP_LAT_PKT_TRANS_DL NUMBER(38,2),
	PM_PDCP_LAT_TIME_UL VARCHAR(16777216),
	PM_PDCP_LAT_PKT_TRANS_UL VARCHAR(16777216),
	PM_UE_THP_TIME_DL NUMBER(38,2),
	PM_PDCP_VOL_DL_DRB NUMBER(38,2),
	PM_PDCP_VOL_DL_DRB_LAST_TTI NUMBER(38,2),
	PM_UE_MEAS_RSRP_DELTA_INTRA_FREQ1 NUMBER(38,0),
	PM_UE_MEAS_RSRP_SERV_INTRA_FREQ1 NUMBER(38,0),
	PM_UE_MEAS_RSRQ_DELTA_INTRA_FREQ1 NUMBER(38,0),
	PM_UE_MEAS_RSRQ_SERV_INTRA_FREQ1 NUMBER(38,0),
	PM_ERAB_REL_ABNORMAL_ENB_ACT NUMBER(38,2),
	PM_ERAB_REL_ABNORMAL_ENB NUMBER(38,2),
	PM_ERAB_REL_NORMAL_ENB NUMBER(38,2),
	PM_ERAB_REL_MME NUMBER(38,2),
	PM_RRC_CONN_ESTAB_SUCC NUMBER(38,2),
	PM_RRC_CONN_ESTAB_ATT NUMBER(38,2),
	PM_RRC_CONN_ESTAB_ATT_REATT NUMBER(38,2),
	PM_S1_SIG_CONN_ESTAB_SUCC NUMBER(38,2),
	PM_S1_SIG_CONN_ESTAB_ATT NUMBER(38,2),
	PM_ERAB_ESTAB_SUCC_INIT NUMBER(38,2),
	PM_ERAB_ESTAB_ATT_INIT NUMBER(38,2),
	PM_PRB_UTIL_DL NUMBER(38,2),
	PM_PRB_UTIL_UL NUMBER(38,2),
	UNIQUE_ID VARCHAR(16777216)
);

-- Copy data into the COUNTRY table from the S3 stage
COPY INTO TELCO_NETWORK_OPTIMIZATION_PROD.RAW.CELL_TOWER
FROM @TELCO_NETWORK_OPTIMIZATION_PROD.RAW.S3LOAD_TELCO_NW_OPT/cell_tower/;


-- Create or replace the SUPPORT_TICKETS table in the RAW schema
create or replace TABLE TELCO_NETWORK_OPTIMIZATION_PROD.RAW.SUPPORT_TICKETS (
	TICKET_ID VARCHAR(60),
	CUSTOMER_NAME VARCHAR(60),
	CUSTOMER_EMAIL VARCHAR(60),
	SERVICE_TYPE VARCHAR(60),
	REQUEST VARCHAR(16777216),
	CONTACT_PREFERENCE VARCHAR(60),
	CELL_ID NUMBER(38,0)
);

-- Copy data into the SUPPORT_TICKETS table from the S3 stage
COPY INTO TELCO_NETWORK_OPTIMIZATION_PROD.RAW.SUPPORT_TICKETS
FROM @TELCO_NETWORK_OPTIMIZATION_PROD.RAW.S3LOAD_TELCO_NW_OPT/support_tickets/;


-- Create or replace the CUSTOMER_LOYALTY table in the RAW schema
create or replace TABLE TELCO_NETWORK_OPTIMIZATION_PROD.RAW.CUSTOMER_LOYALTY (
	ID NUMBER(19,0),
	FIRST_NAME VARCHAR(16777216),
	LAST_NAME VARCHAR(16777216),
	EMAIL VARCHAR(16777216),
	GENDER VARCHAR(16777216),
	STATUS VARCHAR(16777216),
	ADDRESS VARCHAR(16777216),
    	PHONE_NUMBER NUMBER(38,0),
    	POINTS NUMBER(19,0)
);

-- Copy data into the CUSTOMER_LOYALTY table from the S3 stage
COPY INTO TELCO_NETWORK_OPTIMIZATION_PROD.RAW.CUSTOMER_LOYALTY
FROM @TELCO_NETWORK_OPTIMIZATION_PROD.RAW.S3LOAD_TELCO_NW_OPT/customer_loyalty/;

-- let's calculate Sentiment score for the call transcripts using CORTEX SENTIMENT function
ALTER TABLE TELCO_NETWORK_OPTIMIZATION_PROD.RAW.SUPPORT_TICKETS ADD COLUMN sentiment_score FLOAT;

UPDATE TELCO_NETWORK_OPTIMIZATION_PROD.RAW.SUPPORT_TICKETS
SET sentiment_score = SNOWFLAKE.CORTEX.SENTIMENT(request);

-- Grant privileges on the DATABASE to the ANALYST role
GRANT ALL ON DATABASE TELCO_NETWORK_OPTIMIZATION_PROD TO ROLE TELCO_NETWORK_OPTIMIZATION_ANALYST;

-- Grant privileges on SCHEMA to the ANALYST role
GRANT ALL ON SCHEMA TELCO_NETWORK_OPTIMIZATION_PROD.RAW TO ROLE TELCO_NETWORK_OPTIMIZATION_ANALYST;

-- Grant privileges on ALL TABLES in SCHEMA to the ANALYST role
GRANT ALL ON ALL TABLES IN SCHEMA TELCO_NETWORK_OPTIMIZATION_PROD.RAW TO ROLE TELCO_NETWORK_OPTIMIZATION_ANALYST;

-- Grant privileges on the WAREHOUSE to the ANALYST role
GRANT ALL ON WAREHOUSE TELCO_NETWORK_OPTIMIZATION_ANALYST_WH TO ROLE TELCO_NETWORK_OPTIMIZATION_ANALYST;

-- Create stage to host Cortex Analyst Semantic Model yaml file
CREATE STAGE IF NOT EXISTS TELCO_NETWORK_OPTIMIZATION_PROD.RAW.DATA DIRECTORY = (ENABLE = TRUE) ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
GRANT READ, WRITE ON STAGE TELCO_NETWORK_OPTIMIZATION_PROD.RAW.data TO ROLE TELCO_NETWORK_OPTIMIZATION_ANALYST;
