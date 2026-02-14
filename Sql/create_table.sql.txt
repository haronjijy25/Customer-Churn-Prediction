-- =====================================================
-- CUSTOMER CHURN PROJECT — TABLE STRUCTURE
-- =====================================================

CREATE DATABASE IF NOT EXISTS churn_project;
USE churn_project;

CREATE TABLE IF NOT EXISTS customers (
    CustomerID INT PRIMARY KEY,
    Age INT,
    Tenure INT,
    PurchaseFrequency INT,
    AvgOrderValue DECIMAL(10,2),
    TotalSpend DECIMAL(12,2),
    LastPurchaseDaysAgo INT,
    Complaints INT,
    DiscountUsage TINYINT,
    Region VARCHAR(10),
    Churn TINYINT
);
