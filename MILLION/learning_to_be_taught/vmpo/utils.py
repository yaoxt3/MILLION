import torch

def batched_quadratic_form(vector, matrix):
    assert vector.shape[-1] == matrix.shape[-2] == matrix.shape[-1], 'received invalid shapes ' + str(
        vector.shape) + str(matrix.shape)

    result = torch.matmul(
        torch.matmul(vector.unsqueeze(-2), matrix)
        , vector.unsqueeze(-1))
    return result.squeeze(-1).squeeze(-1)

def batched_trace(matrix):
    trace = torch.sum(torch.diagonal(matrix, dim1=-2, dim2=-1), dim=-1)
    return trace
