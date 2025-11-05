KAFKA_DIR='/Users/estefania/Documents/kafka/kafka_2.13-3.9.1'
PROJECT_DIR='/Users/estefania/Documents/workshop-3'


cd $KAFKA_DIR

./bin/zookeeper-server-start.sh -daemon ./config/zookeeper.properties

./bin/kafka-server-start.sh -daemon ./config/server.properties

./bin/kafka-topics.sh --create --if-not-exists \
  --topic happiness_features \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1


cd $PROJECT_DIR

python3.12 scripts/consumer.py 
python3.12 scripts/producer.py