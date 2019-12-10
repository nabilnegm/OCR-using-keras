# OCR-using-keras
Arabic characters and numbers OCR using keras.
we have a folder of 53760 arabic image each representing an arabic character. we also added mnist dataset for the digits.

the layout of the CNN looks like this:
conv=>maxpool=>dropout=>norm=>conv=>maxpool=>dropout=>norm=>flatten=>fullyconnected=>dropout=>norm=>fullyconnected=>softmax
