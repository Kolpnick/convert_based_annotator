curl -H "Content-Type: application/json" -XPOST \
 -d '{"candidates": ["I am cooking", "Am I walking?", "I was there yesterday", "I am in London", "I love Moscow"], "history": []}' \
  http://172.17.0.2:8021/conv_annot_candidate