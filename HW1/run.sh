mkdir -p logs
CONFIG_NAME=cloud_config
LOG_FILE=logs/${CONFIG_NAME}_experts_advice.log
python experts.py $CONFIG_NAME > $LOG_FILE
echo "==========================================================" >> $LOG_FILE
python experts_advice.py $CONFIG_NAME >> $LOG_FILE

CONFIG_NAME=spambase_config
LOG_FILE=logs/${CONFIG_NAME}_experts_advice.log
python experts.py $CONFIG_NAME > $LOG_FILE
echo "==========================================================" >> $LOG_FILE
python experts_advice.py $CONFIG_NAME >> $LOG_FILE 
