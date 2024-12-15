const mysql = require('mysql');
const XLSX = require('xlsx');
const fs = require('fs');

// MySQL connection setup
const connection = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'ariba0113',
    database: 'plantAdvisor'
});

// Read the Excel file
const filePath = 'E:\\ai\\ai lab conference\\plants dataset.xlsx';  // Change this to your file path
const workbook = XLSX.readFile(filePath);

// Parse the first sheet
const sheetName = workbook.SheetNames[0]; // You can choose the sheet by its index or name
const sheet = workbook.Sheets[sheetName];

// Convert the sheet to JSON (each row will be an object)
const data = XLSX.utils.sheet_to_json(sheet);

// Function to insert data into MySQL
const insertData = (plantData) => {
    const query = 'INSERT INTO plants (name, soil_type, fertilization, watering, sunlight) VALUES (?, ?, ?, ?, ?)';

    connection.query(query, [plantData.name, plantData.soil_type, plantData.fertilization, plantData.watering, plantData.sunlight], (err, result) => {
        if (err) {
            console.error('Error inserting data:', err);
        } else {
            console.log('Data inserted successfully:', result);
        }
    });
};

// Loop through the Excel data and insert each row into the database
data.forEach((row) => {
    const plantData = {
        name: row['name'], // Adjust the column names according to your Excel file
        soil_type: row['soil_type'],
        fertilization: row['fertilization'],
        watering: row['watering'],
        sunlight: row['sunlight']
    };

    insertData(plantData);
});

// Close the MySQL connection after all data is inserted
connection.end();
