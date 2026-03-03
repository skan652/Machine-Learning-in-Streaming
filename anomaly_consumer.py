#!/usr/bin/env python
"""Phase 2: Anomaly Detection - Pure Python Kafka Consumer"""
import json
from collections import deque
from kafka import KafkaConsumer
import numpy as np

def main():
    """Consume transactions from Kafka and detect anomalies"""
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id='anomaly-detection-group'
    )
    
    print("✓ Connected to Kafka topic 'transactions'")
    print("✓ Anomaly detection started. Listening for transactions...\n")
    
    # Keep rolling window of amounts
    amounts = deque(maxlen=100)
    batch_count = 0
    transaction_count = 0
    
    for message in consumer:
        transaction = message.value
        transaction_count += 1
        amount = float(transaction.get('Amount', 0))
        amounts.append(amount)
        
        # Every 50 transactions, compute statistics and detect anomalies
        if transaction_count % 50 == 0:
            batch_count += 1
            if len(amounts) > 10:
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts)
                
                # Find which transactions in this batch are anomalies (z-score > 3)
                if std_amount > 0:
                    z_score = abs((amount - mean_amount) / std_amount)
                    if z_score >= 3.0:
                        print(f"\n==== ANOMALY DETECTED (Batch {batch_count}) ====")
                        print(f"  Time:     {transaction.get('Time')}")
                        print(f"  Amount:   ${amount:.2f}")
                        print(f"  Z-score:  {z_score:.2f}")
                        print(f"  Mean:     ${mean_amount:.2f}")
                        print(f"  Std Dev:  ${std_amount:.2f}")
                        print(f"  Label:    {'FRAUD' if transaction.get('Class') == 1 else 'NORMAL'}")
                
                # Print batch status
                print(f"[{transaction_count} events] Mean: ${mean_amount:.2f}, Std: ${std_amount:.2f}")

if __name__ == "__main__":
    main()
