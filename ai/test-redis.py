import redis
import json

print(f"[DEBUG] Test Redis:")
r = redis.Redis(host='localhost', port=6379, db=0)
print(f"[DEBUG] Connected: {r}")
message = {
    'medsession_id': '95'
}
r.publish('medsession-channel', json.dumps(message))
print(f"[DEBUG] publish: json.dumps(message)")