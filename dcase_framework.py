class EventRoll(object):
    def __init__(self, metadata_container, label_list=None, time_resolution=0.01, label='event_label', length=None):
        """Event roll

        Event roll is binary matrix indicating event activity withing time segment defined by time_resolution.

        Parameters
        ----------
        metadata_container : MetaDataContainer
            Meta data
        label_list : list
            List of labels in correct order
        time_resolution : float > 0.0
            Time resolution used when converting event into event roll.
            Default value "0.01"
        label : str
            Meta data field used to create event roll
            Default value "event_label"
        length : int, optional
            length of event roll, if none given max offset of the meta data is used.
            Default value "None"

        """

        self.metadata_container = metadata_container

        if label_list is None:
            self.label_list = metadata_container.unique_event_labels
        else:
            self.label_list = label_list

        self.time_resolution = time_resolution
        self.label = label

        if length is None:
            self.max_offset_value = metadata_container.max_event_offset
        else:
            self.max_offset_value = length

        # Initialize event roll
        self.event_roll = numpy.zeros(
            (int(math.ceil(self.max_offset_value * 1.0 / self.time_resolution)), len(self.label_list))
        )

        # Fill-in event_roll
        for item in self.metadata_container:
            if item.event_onset is not None and item.event_offset is not None:
                if item[self.label]:
                    pos = self.label_list.index(item[self.label])
                    onset = int(numpy.floor(item.event_onset * 1.0 / self.time_resolution))
                    offset = int(numpy.ceil(item.event_offset * 1.0 / self.time_resolution))

                    if offset > self.event_roll.shape[0]:
                        # we have event which continues beyond max_offset_value
                        offset = self.event_roll.shape[0]

                    if onset <= self.event_roll.shape[0]:
                        # We have event inside roll
                        self.event_roll[onset:offset, pos] = 1

    @property
    def roll(self):
        return self.event_roll

    def pad(self, length):
        """Pad event roll's length to given length

        Parameters
        ----------
        length : int
            Length to be padded

        Returns
        -------
        event_roll: np.ndarray, shape=(m,k)
            Padded event roll

        """

        if length > self.event_roll.shape[0]:
            padding = numpy.zeros((length-self.event_roll.shape[0], self.event_roll.shape[1]))
            self.event_roll = numpy.vstack((self.event_roll, padding))

        elif length < self.event_roll.shape[0]:
            self.event_roll = self.event_roll[0:length, :]

        return self.event_roll

    def plot(self):
        """Plot Event roll

        Returns
        -------
        None

        """

        import matplotlib.pyplot as plt
        plt.matshow(self.event_roll.T, cmap=plt.cm.gray, interpolation='nearest', aspect='auto')
        plt.show()
