import sqlite3

def insert_sample_data(sqlite_db):
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    # Sample data
    maintenance_data = [
        (1, 'BS-001', '2023-06-15', 'John Doe', 'Routine check-up', 2, 150.00),
        (2, 'BS-002', '2023-07-10', 'Jane Smith', 'Antenna alignment', 3, 200.00),
        (3, 'BS-003', '2023-07-20', 'Emily Johnson', 'Software update', 1.5, 100.00),
        (4, 'BS-004', '2023-08-05', 'Michael Brown', 'Power supply repair', 4, 300.00),
        (5, 'BS-005', '2023-08-25', 'Sarah Davis', 'Signal interference fix', 2.5, 250.00),
    ]

    # Insert sample data into the BaseStationMaintenance table
    insert_query = """
    INSERT INTO BaseStationMaintenance (Maintenance_ID, Base_Station_ID, Maintenance_Date, Technician_Name, Description, Duration_hours, Cost)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(insert_query, maintenance_data)

    # Commit changes and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    sqlite_db = "data/sample.db"  # Replace with your SQLite database file path
    insert_sample_data(sqlite_db)
    print("Sample data inserted into BaseStationMaintenance table.")