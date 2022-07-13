"""
Last modified Apr 13 2020
@author: Jakub Maly
"""
import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import root_load.root_converter as converter
import warnings
import re
import pickle
import logging
class Reader:
    """ Class for handling pkl files.
    This class is used to open pkl files, get basic information about their structure,
    and obtain their data.
    Please see converter.py for information about format in which data are saved.
    """
    def __init__(self, params={}):
        self.verbose = params.get('verbose')
        self.state = 0
        self.file = {}
        self.data = {}
        self.filename = ''
        self.branches = []
        self.query = ''
        self.criterion = ''
        self.logger = logging.getLogger()
    def open(self, f):
        """ Open specified pkl file.
        @param self:
        @param f: The pkl file which is to be open.
        @return:
        """
        if self.verbose is not None:
            self.logger.info('reader.open():')
            self.logger.info('\t|-> opening file \'' + f + '\'')
        self.file = f
        self.filename = f.rsplit('/', 1)[1][:-4]
        with open(self.file, 'rb') as f:
            self.logger.info("opening file {}".format(f))
            self.data = pickle.load(f)
        # Set criterion
        self.criterion = self.data['criterion012345']
        del self.data['criterion012345']
        # Set branches
        temp = []
        for key in self.data:
            temp.append(key)
        self.branches = temp
        # Set query
        temp = ''
        for item in self.branches:
            if temp:
                temp += ':' + item
            else:
                temp += item
        self.query = temp
        self.state = 1
        if self.verbose is not None:
            self.logger.info('\t--------- open done ----------')
    def print_info(self):
        """ Show basic information about the file.
        @param self:
        @return:
        """
        if self.state < 1:
            warnings.warn('Warning: File need to be opened first! Use method open()')
            return
        if self.verbose is not None:
            self.logger.info('reader.print_info():')
        if self.criterion:
            self.logger.info('\t|-> file \'{0}.pkl\' contains data of \'Scan(\"{1}\",\"{2}\")\''.format(self.filename, self.query,
                                                                                             self.criterion))
        else:
            self.logger.info('\t|-> file \'{0}.pkl\' contains data of \'Scan(\"{1}\")\''.format(self.filename, self.query))
    def filt_channels(self, features, channels):
        """ Filter data by given features, ignore desired channels.
        @param self:
        @param features: Features used for filtering.
        @param channels: Channels used for filtering
        @return:
        """
        if self.state < 1:
            warnings.warn('Warning: File need to be opened first! Use method open()')
            return
        if self.verbose is not None:
            self.logger.info('reader.filt_channels():')
        # Wildcard handling
        temp_features = list()
        for feature in features:
            if feature[-1] == "*":
                for element in filter((lambda x: re.search(r'^{}'.format(feature[:-1]), x)), self.branches):
                    temp_features.append(element)
            else:
                temp_features.append(feature)
        for channel in channels:
            temp_features.append(channel)
        # Missing feature handling
        temp_features_checked = list()
        for feature in temp_features:
            if feature in self.branches:
                temp_features_checked.append(feature)
            else:
                self.logger.info("Removing missing feature -" , feature)
        self.data = {feature: self.data[feature] for feature in temp_features_checked}
    def filt_criterion(self, features, criterion):
        """ Filter data by given features, applies desired criterion
        @param self:
        @param features: Features used for filtering.
        @param criterion: Criterion used for filtering.
        @return:
        """
        if self.state < 1:
            warnings.warn('Warning: File need to be opened first! Use method open()')
            return
        if self.verbose is not None:
            self.logger.info('reader.filt_criterion():')
        if criterion is None:
            # Wildcard handling
            temp_features = list()
            for feature in features:
                if feature[-1] == "*":
                    for element in filter((lambda x: re.search(r'^{}'.format(feature[:-1]), x)), self.branches):
                        temp_features.append(element)
                else:
                    temp_features.append(feature)
            # Missing feature handling
            temp_features_checked = list()
            for feature in temp_features:
                if feature in self.branches:
                    temp_features_checked.append(feature)
            self.data = {feature: self.data[feature] for feature in temp_features_checked}
        else:
            cv = converter.Converter()
            criterion, req_branches = cv.to_code(criterion)
            if self.verbose is not None:
                self.logger.info('\t------------------------------')
                self.logger.info('\t|-> applying criterion \'' + criterion + '\'')
            dc = self.data
            result = eval(criterion)
            for key in dc:
                dc[key] = dc[key][result]
            self.data = dc
            self.filt_criterion(features, None)
    def save(self, f):
        """ Save data in pkl format.
        @param self:
        @param f: Path to file
        @return:
        """
        self.data['criterion012345'] = self.criterion
        with open(f, 'wb') as f:
            pickle.dump(self.data, f)
            f.close()
    def get_data(self):
        """ Return file's data.
        @param self:
        @return: Dictionary with data.
        """
        if self.state < 1:
            warnings.warn('Warning: File need to be opened first! Use method open()')
        return self.data
    def get_branches(self):
        """ Return file's branches.
        @param self:
        @return: List of branches names.
        """
        if self.state < 1:
            warnings.warn('Warning: File need to be opened first! Use method open()')
        return self.branches
    def get_query(self):
        """ Return query string.
        @param self:
        @return: Query used for Scan() command.
        """
        if self.state < 1:
            warnings.warn('Warning: File need to be opened first! Use method open()')
        return self.query
    def get_criterion(self):
        """ Return criterion string.
        @param self:
        @return: Criterion used for Scan() command.
        """
        if self.state < 1:
            warnings.warn('Warning: File need to be opened first! Use method open()')
        return self.criterion
