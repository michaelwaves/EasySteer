import json
import re
import csv
import random

data = []
random.seed(42)

def parse_vast_offer(item):
    """Parse a Vast.AI offer and return dict or None"""
    try:
        num_gpus = item.get('num_gpus', 1)
        vram = item.get('gpu_ram', 0)  # VRAM in MB
        cost_per_hour = item.get('dph_total', 0)  # cost per hour for all GPUs

        # Normalize cost to single GPU
        if num_gpus > 0:
            cost_per_gpu_hour = cost_per_hour / num_gpus
        else:
            cost_per_gpu_hour = cost_per_hour

        return {
            'vram': round(vram / 1024, 1),  # Convert MB to GB
            'gpu': item.get('gpu_name', 'Unknown'),
            'cost_per_hour': round(cost_per_gpu_hour, 4),
            'provider': 'Vast.AI',
            'source': 'https://cloud.vast.ai/'
        }
    except:
        return None

# Process vast.jsonl - ALL entries
print("Processing vast.jsonl...")
vast_jsonl_data = []
with open('vast.jsonl', 'r') as f:
    content = f.read()
    objects_text = re.split(r'(?<=\}),\n(?=\{)', content)

    for obj_text in objects_text:
        obj_text = obj_text.strip()
        if not obj_text:
            continue
        try:
            item = json.loads(obj_text)
            parsed = parse_vast_offer(item)
            if parsed:
                vast_jsonl_data.append(parsed)
        except json.JSONDecodeError:
            pass

data.extend(vast_jsonl_data)
print(f"Vast.AI (from vast.jsonl): {len(vast_jsonl_data)} entries (all)")

# Process vast_2.json - SAMPLE 10
print("Processing vast_2.json...")
vast_2_data = []
try:
    with open('vast_2.json', 'r') as f:
        vast2_full = json.load(f)
        offers = vast2_full.get('offers', [])

        # Parse all, then sample
        parsed_offers = [parse_vast_offer(item) for item in offers]
        parsed_offers = [o for o in parsed_offers if o is not None]

        # Sample 10 randomly
        sampled = random.sample(parsed_offers, min(10, len(parsed_offers)))
        vast_2_data.extend(sampled)
except FileNotFoundError:
    print("vast_2.json not found, skipping")

data.extend(vast_2_data)
print(f"Vast.AI (from vast_2.json): {len(vast_2_data)} entries (sampled from {len(parsed_offers)})")

# Process vast_3.json - SAMPLE 10
print("Processing vast_3.json...")
vast_3_data = []
try:
    with open('vast_3.json', 'r') as f:
        vast3_full = json.load(f)
        offers = vast3_full.get('offers', [])

        # Parse all, then sample
        parsed_offers = [parse_vast_offer(item) for item in offers]
        parsed_offers = [o for o in parsed_offers if o is not None]

        # Sample 10 randomly
        sampled = random.sample(parsed_offers, min(10, len(parsed_offers)))
        vast_3_data.extend(sampled)
except FileNotFoundError:
    print("vast_3.json not found, skipping")

data.extend(vast_3_data)
print(f"Vast.AI (from vast_3.json): {len(vast_3_data)} entries (sampled from {len(parsed_offers) if vast_3_data else 0})")

vast_total = len([d for d in data if d['provider'] == 'Vast.AI'])
print(f"Vast.AI (total): {vast_total} entries")

# Process runpod.jsonl - array without opening bracket
print("Processing runpod.jsonl...")
with open('runpod.jsonl', 'r') as f:
    content = f.read()
    # Wrap with brackets to make valid JSON array
    wrapped = '[' + content + ']'
    try:
        items = json.loads(wrapped)
        for item in items:
            if 'memoryInGb' in item and 'securePrice' in item:
                vram = item.get('memoryInGb', 0)
                cost_per_hour = item.get('securePrice', 0)

                data.append({
                    'vram': vram,
                    'gpu': item.get('displayName', item.get('id', 'Unknown')),
                    'cost_per_hour': round(cost_per_hour, 4),
                    'provider': 'RunPod',
                    'source': 'https://console.runpod.io/deploy'
                })
    except json.JSONDecodeError as e:
        print(f"Error parsing runpod: {e}")

print(f"RunPod: {len([d for d in data if d['provider'] == 'RunPod'])} entries")

# Process lambda.json
print("Processing lambda.json...")
with open('lambda.json', 'r') as f:
    items = json.load(f)
    for key, item in items.items():
        instance_type = item.get('instance_type', {})
        specs = instance_type.get('specs', {})

        num_gpus = specs.get('gpus', 1)

        # Extract VRAM from description
        desc = instance_type.get('gpu_description', '')
        vram_in_gb = 0
        if 'GB' in desc:
            parts = desc.split('(')
            if len(parts) > 1:
                vram_str = parts[1].split('GB')[0].strip()
                try:
                    vram_in_gb = int(vram_str)
                except:
                    pass

        cost_cents = instance_type.get('price_cents_per_hour', 0)
        cost_per_hour = cost_cents / 100.0  # Convert cents to dollars

        # Normalize to single GPU
        if num_gpus > 0:
            cost_per_gpu_hour = cost_per_hour / num_gpus
        else:
            cost_per_gpu_hour = cost_per_hour

        data.append({
            'vram': vram_in_gb,
            'gpu': instance_type.get('gpu_description', 'Unknown'),
            'cost_per_hour': round(cost_per_gpu_hour, 4),
            'provider': 'Lambda Labs',
            'source': 'https://cloud.lambda.ai/instances'
        })

print(f"Lambda: {len([d for d in data if d['provider'] == 'Lambda Labs'])} entries")

# Write CSV
print(f"\nWriting {len(data)} entries to output.csv...")
with open('output.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['vram', 'gpu', 'cost_per_hour', 'provider', 'source'])
    writer.writeheader()
    writer.writerows(data)

print("Done!")
print(f"\nCSV created with {len(data)} GPU entries\n")

# Show breakdown
print("Breakdown by provider:")
for provider in ['Vast.AI', 'RunPod', 'Lambda Labs']:
    count = len([d for d in data if d['provider'] == provider])
    if count > 0:
        provider_data = [d for d in data if d['provider'] == provider]
        costs_nonzero = [d['cost_per_hour'] for d in provider_data if d['cost_per_hour'] > 0]
        if costs_nonzero:
            min_cost = min(costs_nonzero)
            max_cost = max(costs_nonzero)
            print(f"  {provider}: {count} entries (${min_cost:.4f} - ${max_cost:.4f}/hr)")
        else:
            print(f"  {provider}: {count} entries (all free tier)")
