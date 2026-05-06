import requests, re
r = requests.get('http://127.0.0.1:5000/')
html = r.text
m = re.search(r'<div class="brain-wrap">(.*?)</div>', html, re.S)
if m:
    inner = m.group(1)
    print('FOUND', len(inner))
    print(inner)
else:
    print('brain-wrap not found')
