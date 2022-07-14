from harvesters.core import Harvester
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

h = Harvester()

print(h.add_file('C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti'))
h.update()
print(h.device_info_list)

ia = h.create(0)

ia.remote_device.node_map.Width.value = 2448

ia.remote_device.node_map.Height.value = 2048

ia.remote_device.node_map.PixelFormat.value = 'BayerRG8'

ia.start()

with ia.fetch() as buffer:
    component = buffer.payload.components[0]
    _1d = component.data
    print('1D: {0}'.format(_1d))
    _2d = component.data.reshape(
        component.height, component.width
    )
    print('2D: {0}'.format(_2d))
    assert isinstance(_2d, object)
    imgplot = plt.imshow(_2d)
    plt.show()
    # Here are some trivial calculations:
    print(
        'AVE: {0}, MIN: {1}, MAX: {2}'.format(
            np.average(_2d), _2d.min(), _2d.max()
        )
    )

# Stop the ImageAcquier object acquiring images:
ia.stop()

# We're going to leave here shortly:
ia.destroy()
h.reset()
