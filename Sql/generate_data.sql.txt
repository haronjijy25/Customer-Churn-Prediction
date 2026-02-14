-- =====================================================
-- CUSTOMER CHURN PROJECT — DATA GENERATION
-- =====================================================

DELIMITER $$

CREATE PROCEDURE generate_customers()
BEGIN
    DECLARE i INT DEFAULT 1;

    WHILE i <= 1000 DO

        INSERT INTO customers VALUES (
            i,

            FLOOR(18 + RAND()*53),                 -- Age
            FLOOR(1 + RAND()*60),                  -- Tenure
            FLOOR(1 + RAND()*20),                  -- Purchase Frequency
            ROUND(200 + RAND()*4800, 2),           -- Avg Order Value

            ROUND((200 + RAND()*4800) *
                  (1 + RAND()*20) *
                  (1 + RAND()*5), 2),              -- Total Spend

            FLOOR(RAND()*365),                     -- Recency
            FLOOR(RAND()*6),                       -- Complaints
            FLOOR(RAND()*2),                       -- Discount Usage

            ELT(FLOOR(1 + RAND()*4), 'North','South','West','East'),

            CASE
                WHEN RAND() < 0.20 THEN 1
                WHEN RAND() < 0.35 THEN 1
                ELSE 0
            END
        );

        SET i = i + 1;

    END WHILE;
END$$

DELIMITER ;

-- Run this to generate dataset
CALL generate_customers();
