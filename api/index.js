const express = require('express');
const app = express();
const server = require('http').createServer(app); 
const RabbitMQ = require('./config/rabbitmq/index');
const multer = require('multer');
const upload = multer({ dest: './' });
const fs = require('fs');
const cors = require("cors");
const Socket = require('./config/socket-io');
const allowedOrigins = [
  "*",
  "localhost",
  "http://localhost:8080",
  "http://127.0.0.1:8080",
  "http://localhost:8082/",
];
var corsOptions = {
  origin: function (origin, callback) {
    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      // callback(new Error('Not allowed by CORS'))
      // console.log("------")
      // console.log("origin",origin)
      callback(null, true);
    }
  },
  credentials: true,
};
app.use(cors(corsOptions));
app.get('/', (req, res)=>{
    res.sendStatus(200)
})
app.post('/', upload.single('image'), async (req, res) => {
  const file = req.file;
  const contents = await fs.readFileSync(file.path, { encoding: 'base64' });
  RabbitMQ.getInstance().rpc(contents);
  fs.unlinkSync(file.path);
  res.status(200).json({ 'data': 'ok' });
});

const io = Socket.getInstance().createIO(server)
io.on('connection', (socket) => {
    console.log('a user connected');
    socket.on('disconnect', () => {
      console.log('user disconnected');
    });
  });
server.listen(3003, () => {
  console.log('running on port 3003..');
});