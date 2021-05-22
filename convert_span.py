from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np

def convert_head_to_span(all_heads):
    hpsg_lists = []
    for heads in all_heads:
        n = len(heads)
        childs = [[] for i in range(n+1)]
        left_p = [i for i in range(n+1)]
        right_p = [i for i in range(n+1)]

        def dfs(x):
            for child in childs[x]:
                dfs(child)
                left_p[x] = min(left_p[x], left_p[child])
                right_p[x] = max(right_p[x], right_p[child])

        for i, head in enumerate(heads):
            childs[head].append(i+1)

        dfs(0)
        hpsg_list = []
        for i in range(1,n+1):
            hpsg_list.append((left_p[i], right_p[i]))
            
        hpsg_lists.append(hpsg_list)

    return hpsg_lists


def convert_span(example, tokenizer):
    org_sen_token = example["sen_token"]
    all_sen_span = example["sen_span"]
    split_sen_tokens = []
    org_to_split_map = {}
    pre_tok_len = 0
    for idx, token in enumerate(org_sen_token):
        sub_tok = tokenizer.tokenize(token)
        org_to_split_map[idx] = (pre_tok_len, len(sub_tok) + pre_tok_len - 1)
        pre_tok_len += len(sub_tok)
        split_sen_tokens.extend(sub_tok)

    cnt_span = 0
    for sen_idx, sen_span in enumerate(all_sen_span):
        for idx, (start_ix, end_ix) in enumerate(sen_span):
            assert (start_ix <= len(sen_span) and end_ix <= len(sen_span))
            cnt_span += 1
    assert cnt_span == len(org_sen_token)

    sub_sen_span = []
    pre_sen_len = 0
    for sen_idx, sen_span in enumerate(all_sen_span):
        sen_offset = pre_sen_len
        pre_sen_len += len(sen_span)
        for idx, (start_ix, end_ix) in enumerate(sen_span):
            tok_start, tok_end = org_to_split_map[sen_offset+idx]
#             print("tok_start: ", tok_start)
#             print("tok_end: ", tok_end)
#             input()
            # sub_start_idx and sub_end_idx of children of head node
            head_spans = [(org_to_split_map[sen_offset+start_ix-1][0], org_to_split_map[sen_offset+end_ix-1][1])]
#             print("head_spans:", head_spans)
#             input()
            # all other head sub_tok point to first head sub_tok
            if tok_start != tok_end:
#                 head_spans.append((tok_start + 1, tok_end))
                sub_sen_span.append(head_spans)

                for i in range(tok_start + 1, tok_end + 1):
                    sub_sen_span.append([(i, i)])
            else:
                sub_sen_span.append(head_spans)
    print("sub_sen_span: ", sub_sen_span)
    input()
    assert len(sub_sen_span) == len(split_sen_tokens)

    # making masks
    span_mask = np.zeros((len(sub_sen_span), len(sub_sen_span)))

    for idx, span_list in enumerate(sub_sen_span):
        for (start_ix, end_ix) in span_list:
            span_mask[start_ix:end_ix + 1, idx] = 1
    # record_mask records ancestor nodes for each wd
    record_mask = []
    for i in range(len(sub_sen_span)):
        i_mask = []
        for j in range(len(sub_sen_span)):
            if span_mask[i, j] == 1:
                i_mask.append(j)
        record_mask.append(i_mask)

    return split_sen_tokens, span_mask, record_mask


def test():
    print("Start test...")
    sen_tokens = ["The", "legislation", "allowed", "California", "to", "be", "admitted", "to", "the", "Union", "as", "what", "kind", "of", "state?", "The", "legislation", "allowed", "California", "to", "be", "admitted", "to", "the", "Union", "as", "what", "kind", "of", "state?"]
    sen_heads = [[2, 3, 0, 7, 7, 7, 3, 7, 10, 8, 7, 13, 11, 13, 3], [2, 3, 0, 7, 7, 7, 3, 7, 10, 8, 7, 13, 11, 13, 3]]
    sen_span = [[(1, 1), (1, 2), (1, 15), (4, 4), (5, 5), (6, 6), (4, 14), (8, 10), (9, 9), (9, 10), (11, 14), (12, 12)
    , (12, 14), (14, 14), (15, 15)],[(1, 1), (1, 2), (1, 15), (4, 4), (5, 5), (6, 6), (4, 14), (8, 10), (9, 9), (9, 10), (11, 14), (12, 12)
    , (12, 14), (14, 14), (15, 15)]]
    test_sen_span = convert_head_to_span(sen_heads)
    print("sen_span: ", test_sen_span)
    assert test_sen_span == sen_span
    example = {"sen_token": sen_tokens, "sen_span": sen_span}
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    tokens, input_span_mask, record_mask = convert_span(example, tokenizer)
    print("tokens: ", tokens)
    print("span_mask: ", input_span_mask)
    print("record_mask: ", record_mask)
    

if __name__ == "__main__":
    test()

