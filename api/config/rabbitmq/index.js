const amqp = require("amqplib/callback_api");
const Socket = require("../socket-io");
const io = Socket.getInstance().io

const callback = (msg)=>{
  console.log(msg)
}

class RabbitMQ {
  instance = null;
  constructor() {}
  static getInstance() {
    if (!this.instance) {
      this.instance = new RabbitMQ();
    }
    return this.instance;
  }
  createExchange(name, mode) {
    amqp.connect("amqp://localhost", (err0, connection) => {
      if (err0) console.log(err0);
      else
        connection.createChannel((err1, channel) => {
          if (err1) console.log(err1);
          else channel.assertExchange(name, mode);
        });
      setTimeout(() => {
        connection.close();
      }, 500);
    });
  }
  send(exchange, msg) {
    amqp.connect("amqp://localhost", (err0, connection) => {
      if (err0) console.log(err0);
      else
        connection.createChannel((err1, channel) => {
          if (err1) console.log(err1);
          else {
            channel.publish(exchange, "", Buffer.from(msg));
            console.log(" [x] send %s", msg);
          }
        });
      setTimeout(() => {
        connection.close();
      }, 500);
    });
  }
  rpc(msg) {
    amqp.connect("amqp://localhost", async (err0, connection) => {
      connection.createChannel((err1, channel) => {
        const generateUuid = () => {
          return (
            Math.random().toString() +
            Math.random().toString() +
            Math.random().toString()
          );
        };
        const queue = "rpc_queue";
        channel.assertQueue("", { exclusive: true }, (err2, replyQueue) => {
          channel.consume(
            replyQueue.queue,
            (msg) => {
              if (msg.properties.correlationId == correlationId) {
                callback(msg.content.toString())
                setTimeout(() => {
                  connection.close();
                }, 500);
              }
            },
            { noAck: true }
          );
          const correlationId = generateUuid();
          channel.sendToQueue(queue, Buffer.from(msg), {
            correlationId,
            replyTo: replyQueue.queue,
          });
          console.log(`Sending RPC request...\n${correlationId}`);
        });
      });
    });
  }
}
module.exports = RabbitMQ;
