// Load Node modules
var express = require('express');
const ejs = require('ejs');
const path = require('path/posix');
const request = require('request');

// Initialise Express
var app = express();
// Render static files
app.use(express.static('static'));
// Set the view engine to ejs
app.set('view engine', 'ejs');

port = "8080"

// Port website will run on
app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`)
})

// *** GET Routes - display pages ***
// Root Route
app.get('/', function (req, res) {
    request('http://127.0.0.1:5000/flask', function (error, response, body) {
        console.error('error:', error); // Print the error
        console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
        // console.log('body:', body); // Print the data received
        res.send(body); //Display the response on the website
    });   
});