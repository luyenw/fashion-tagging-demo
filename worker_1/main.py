import pika
import time
import tensorflow as tf
from utils import process_one_image, get_response_msg
# load tf model
print('Loading Model ...')
load_start = time.time()
model = tf.saved_model.load('mask_rcnn_model')
load_end = time.time()
# redis 
import redis
from Crypto.Hash import SHA256
_redis = redis.Redis(host='localhost', port=6378, decode_responses=True)
# rabbitmq 
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='rpc_queue')
import cv2 
def fibonacci(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prediction = process_one_image(model, img)
    msg = get_response_msg(prediction)
    return msg

import os
import base64
import hashlib
import threading 

def on_request(ch, method, props, body):
    print("Awaiting RPC requests...")
    def process_request():
        begin = time.time()
        hash_object = hashlib.sha256(body).hexdigest()
        if _redis.hgetall(hash_object):
            output = _redis.hgetall(hash_object)
        else:
            n = body.decode()
            contents = base64.b64decode(n)
            filename = f'{props.correlation_id}.jpg' 
            with open(filename, 'wb') as f:
                f.write(contents)
            output = fibonacci(filename)
            _redis.hset(hash_object, mapping=output)
            os.remove(filename)
        output['from'] = 'worker_1'
        output['target_id'] = props.correlation_id
        end = time.time()
        print(f"Sending response: ok {end-begin}")
        ch.basic_publish(
            exchange='',
            routing_key=str(props.reply_to),
            properties=pika.BasicProperties(correlation_id=props.correlation_id),
            body=str(output)
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)
    thread = threading.Thread(target=process_request)
    thread.start()


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)
channel.start_consuming()