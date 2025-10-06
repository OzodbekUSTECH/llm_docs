-- ============================================================
-- PostgreSQL Database Tasks - EMPLOYEE Table (Advanced Modifications)
-- ============================================================

-- Task 1: Assume DEPARTMENT table exists for FK references
DROP TABLE IF EXISTS DEPARTMENT CASCADE;
CREATE TABLE IF NOT EXISTS DEPARTMENT (
    Id INTEGER PRIMARY KEY,
    Name VARCHAR(100) NOT NULL
);

-- Task 2: Clean up EMPLOYEE and sequence if exist
DROP TABLE IF EXISTS EMPLOYEE CASCADE;
DROP SEQUENCE IF EXISTS employee_id_seq;

-- Task 12: Create sequence for Id (PK), start from 1, increment by 5
CREATE SEQUENCE employee_id_seq
    START WITH 1
    INCREMENT BY 5
    MINVALUE 1;

-- Task 3,4,8,9,10,11,12: Create EMPLOYEE table with all required constraints
CREATE TABLE EMPLOYEE (
    Id INTEGER PRIMARY KEY DEFAULT nextval('employee_id_seq'),
    FirstName VARCHAR(25) NOT NULL,
    LastName VARCHAR(60) NOT NULL, -- Task 3: 60 chars, Task 5: will drop/add below
    Dob DATE NOT NULL,             -- Task 4: NOT NULL
    Address VARCHAR(200),
    Email VARCHAR(100) NOT NULL,
    Phone VARCHAR(20),
    Salary NUMERIC(10,2) NOT NULL DEFAULT 1000, -- Task 10: default 1000
    DateEmployed TIMESTAMP NOT NULL DEFAULT CURRENT_DATE,
    DepartmentId INTEGER NOT NULL, -- Task 8: NOT NULL
    CONSTRAINT UQ_Employee_Email UNIQUE (Email), -- Task 6: will drop/add below
    CONSTRAINT CK_Employee_Salary CHECK (Salary > 0 AND Salary <= 5000), -- Task 9
    CONSTRAINT CK_Employee_Email_At CHECK (Email LIKE '%@%'), -- Task 11
    CONSTRAINT FK_Employee_Department FOREIGN KEY (DepartmentId) REFERENCES DEPARTMENT(Id) -- Task 7
);

-- Task 5: Drop and add back LastName column (demonstration)
ALTER TABLE EMPLOYEE DROP COLUMN LastName;
ALTER TABLE EMPLOYEE ADD COLUMN LastName VARCHAR(60) NOT NULL;

-- Task 6: Drop and add back PK and unique constraint for Email
ALTER TABLE EMPLOYEE DROP CONSTRAINT IF EXISTS UQ_Employee_Email;
ALTER TABLE EMPLOYEE DROP CONSTRAINT IF EXISTS EMPLOYEE_pkey;
ALTER TABLE EMPLOYEE ADD CONSTRAINT EMPLOYEE_pkey PRIMARY KEY (Id);
ALTER TABLE EMPLOYEE ADD CONSTRAINT UQ_Employee_Email UNIQUE (Email);

-- Task 7: Drop and re-add FK constraint for DepartmentId
ALTER TABLE EMPLOYEE DROP CONSTRAINT IF EXISTS FK_Employee_Department;
ALTER TABLE EMPLOYEE ADD CONSTRAINT FK_Employee_Department FOREIGN KEY (DepartmentId) REFERENCES DEPARTMENT(Id);

-- Task 13: Insert test data and test constraints

-- Insert departments
INSERT INTO DEPARTMENT (Id, Name) VALUES (1, 'HR');
INSERT INTO DEPARTMENT (Id, Name) VALUES (2, 'IT');

-- Insert employees (valid)
INSERT INTO EMPLOYEE (FirstName, LastName, Dob, Address, Email, Phone, Salary, DepartmentId)
VALUES ('John', 'Smith', '1990-01-01', '123 Main St', 'john.smith@example.com', '1234567890', 2000, 1);

INSERT INTO EMPLOYEE (FirstName, LastName, Dob, Address, Email, Phone, Salary, DepartmentId)
VALUES ('Jane', 'Doe', '1985-05-05', '456 Elm St', 'jane.doe@example.com', '0987654321', 3000, 2);

-- Try to violate PK (should fail)
-- INSERT INTO EMPLOYEE (Id, FirstName, LastName, Dob, Address, Email, Phone, Salary, DepartmentId)
-- VALUES (1, 'Duplicate', 'PK', '1992-02-02', '789 Oak St', 'dup@example.com', '1112223333', 1500, 1);

-- Try to violate unique Email (should fail)
-- INSERT INTO EMPLOYEE (FirstName, LastName, Dob, Address, Email, Phone, Salary, DepartmentId)
-- VALUES ('Sam', 'Unique', '1993-03-03', '101 Pine St', 'john.smith@example.com', '2223334444', 1200, 1);

-- Try to violate salary check (should fail)
-- INSERT INTO EMPLOYEE (FirstName, LastName, Dob, Address, Email, Phone, Salary, DepartmentId)
-- VALUES ('Low', 'Salary', '1994-04-04', '202 Maple St', 'low.salary@example.com', '3334445555', -10, 1);

-- Try to violate FK (should fail)
-- INSERT INTO EMPLOYEE (FirstName, LastName, Dob, Address, Email, Phone, Salary, DepartmentId)
-- VALUES ('Ghost', 'Dept', '1995-05-05', '303 Cedar St', 'ghost.dept@example.com', '4445556666', 1200, 999);

-- Task 14: Try to delete a department referenced by employee (should fail)
-- DELETE FROM DEPARTMENT WHERE Id = 1;

-- To avoid FK error, first delete employees referencing the department, then delete department
DELETE FROM EMPLOYEE WHERE DepartmentId = 1;
DELETE FROM DEPARTMENT WHERE Id = 1;

-- Task 15: Drop and recreate FK with different ON DELETE/UPDATE options

-- CASCADE
ALTER TABLE EMPLOYEE DROP CONSTRAINT IF EXISTS FK_Employee_Department;
ALTER TABLE EMPLOYEE ADD CONSTRAINT FK_Employee_Department
    FOREIGN KEY (DepartmentId) REFERENCES DEPARTMENT(Id)
    ON DELETE CASCADE ON UPDATE CASCADE;

-- Try: Deleting department 2 will delete employees in that department
-- DELETE FROM DEPARTMENT WHERE Id = 2;

-- SET NULL
ALTER TABLE EMPLOYEE DROP CONSTRAINT IF EXISTS FK_Employee_Department;
ALTER TABLE EMPLOYEE ALTER COLUMN DepartmentId DROP NOT NULL;
ALTER TABLE EMPLOYEE ADD CONSTRAINT FK_Employee_Department
    FOREIGN KEY (DepartmentId) REFERENCES DEPARTMENT(Id)
    ON DELETE SET NULL ON UPDATE CASCADE;

-- SET DEFAULT
ALTER TABLE EMPLOYEE DROP CONSTRAINT IF EXISTS FK_Employee_Department;
-- Set default department id to 1 for demonstration
ALTER TABLE EMPLOYEE ALTER COLUMN DepartmentId SET DEFAULT 1;
ALTER TABLE EMPLOYEE ADD CONSTRAINT FK_Employee_Department
    FOREIGN KEY (DepartmentId) REFERENCES DEPARTMENT(Id)
    ON DELETE SET DEFAULT ON UPDATE CASCADE;

-- Verify table creation
SELECT table_name FROM information_schema.tables WHERE table_name = 'employee';
