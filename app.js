const express = require("express");
const session = require("express-session");
const bodyParser = require("body-parser");
const cron = require('node-cron');
const { spawn } = require('child_process');

const cryptoController = require('./controllers/crypto_controller');
const dataFetch = require('./models/data_fetch');
const cronTask = require('./models/cron_task');

const app = express();

app.set("view engine", "ejs");

app.use(bodyParser.urlencoded({extended: true}));
app.use(express.static("public"));
app.use(session({
    secret: "crypto",
    resave: false,
    saveUninitialized: false,
    rolling: true,
    cookie: { maxAge: 300000 }
}));

app.get("/", cryptoController.index);
app.get("/fetch", dataFetch.apiFetch);
app.post("/getCryptoInfo", cryptoController.getCryptoInfo);
app.post("/updateNewsInput", cryptoController.updateNewsInput);
app.post("/rePredict", cryptoController.rePredict);

// cron task right after server is on
// cronTask.predictionTask();

cron.schedule("0 0 * * *", () => {
  // cron task at everyday 12am
  // https://crontab.guru/every-day
  try {
    cronTask.predictionTask();
  } catch (error) {
    console.error(error);
  }
});

app.listen(3000, () =>  {
  console.log("Server listening on port 3000!")
});
