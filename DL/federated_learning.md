# Federated Learning

This seems to be more of Google's vision and advocating for it, haven't done
enough homework to see elsewhere. 

## Vision

Since many data are born at the edge, the vision of federated learning is to
enable the edge devices to participate and help learning the model
while preserving privacy. Privacy is a key consideration here: in the context
the Gboard text prediction, you want the search content and context not to
leave the mobile device, for example.

## Learning on Decentralized Data

The key here: train locally, learn globally. The question is how. A few ideas
flowed around:

* federated aggregate and count: this can be done without revealing individual
  data point.
* federated sum: distributed **data masks** to all participants, and raw data are added on
  top it, when collected at the server side, the masks are cancelled with each
  other, you end up with a valid sum.
* differential privacy: this is another way to obfuscate the origin data,
  where the input was added some noise to intentionally impact the model. It
  is argued that it can be benefitial to the learning while preserving
  privacy.
  
  ## Challenges (vs. centralized learning)

  * data is decentralized
  * drops out from participants
  * limited communication
  * ...
  

There are several papers from Google along this direction. The TensorFlow
Federated provides some examples as well.

