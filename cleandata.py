import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
file_path = 'output.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Assuming the sentiment labels are in a column named 'sentiment'
# Count the sentiment distribution
sentiment_counts = df['sentiment'].value_counts()

# Print sentiment distribution
print("Sentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count}")

# Optional: Plotting the sentiment distribution
sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
