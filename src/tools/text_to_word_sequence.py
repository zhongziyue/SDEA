def text_to_word_sequence(text,
                          filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if text is None:
        return []
    if lower: text = text.lower()
    translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]
