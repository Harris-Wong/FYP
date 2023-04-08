const { spawn } = require('child_process');
const { apiFetch } = require('./data_fetch.js');

function predictionTask() {
  // Run 'Fetch Data' task
  console.log('Task "Fetch Data" process running...');
  const fetchDataProcess = spawn('node', ['-e', `${apiFetch.toString()}; apiFetch()`]);
  fetchDataProcess.stdout.on('data', (data) => {
    console.log(data.toString());
  });
  fetchDataProcess.stderr.on('data', (data) => {
    console.log(data.toString());
  });
  fetchDataProcess.on('close', (code) => {
    console.log(`Task 'Fetch Data' process exited with code ${code}`);
  });

  // Run Python task
  
}

module.exports = { predictionTask };
