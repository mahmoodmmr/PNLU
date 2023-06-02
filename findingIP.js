// let nodeGeocoder = require("node-geocoder");

// let options = {
//   provider: "openstreetmap",
// };

// let geoCoder = nodeGeocoder(options);

// geoCoder
//   .geocode("zanjan")
//   .then((res) => {
//     console.log(res);
//   })
//   .catch((err) => {
//     console.log(err);
//   });

// 2 ------------------------------------------
// var geoip = require("geoip-lite");

// // var ip = "207.97.227.239";
// var ip = "178.131.142.230";
// var geo = geoip.lookup(ip);

// console.log(geo);

// 3 ------------------------------------------
const http = require('http');

http.createServer((req, res) => {
  const ip = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
  const apiUrl = `http://ip-api.com/json/${ip}`;

  http.get(apiUrl, (response) => {
    let data = '';

    response.on('data', (chunk) => {
      data += chunk;
    });

    response.on('end', () => {
      const locationData = JSON.parse(data);

      const city = locationData.city;
      const region = locationData.regionName;
      const country = locationData.country;

      console.log(`User location: ${city}, ${region}, ${country}`);
      res.end();
    });
  });
}).listen(3000);

