from .clocs_second_head import ClocsSECONDHead
from .clocs_dense_head import ClocsDenseHead
from .clocs_voxelrcnn_head import ClocsVoxelRCNNHead
from .clocs_second_contra_head import ClocsSecondContraHead

__all__ = {
    'ClocsVoxelRCNNHead': ClocsVoxelRCNNHead,
    'ClocsSecondContraHead' : ClocsSecondContraHead,
    'ClocsDenseHead': ClocsDenseHead,
    'ClocsSECONDHead': ClocsSECONDHead,
}
