import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

try:
    # Check usage
    usage = client.usage.retrieve()
    print("Usage information:")
    print(f"Total usage: {usage.total_usage}")
    print(f"Total granted: {usage.total_granted}")
    print(f"Total available: {usage.total_available}")
    
    # Check subscription
    subscription = client.billing.subscription.retrieve()
    print("\nSubscription information:")
    print(f"Plan: {subscription.plan}")
    print(f"Status: {subscription.status}")
    print(f"Hard limit: {subscription.hard_limit}")
    print(f"Soft limit: {subscription.soft_limit}")
    
except Exception as e:
    print(f"Error checking quota: {e}")
