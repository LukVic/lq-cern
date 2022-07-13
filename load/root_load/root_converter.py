"""
Last modified Apr 13 2020
@author: Jakub Maly
"""
import logging
from os import path
import root_load.root_reader as reader
import warnings
import uproot3 as uproot
import re
import pickle
import numpy as np


class Converter:
    """ Class for handling root files.

    This class is used to open root files, use the ROOT Scan() method above them,
    convert obtained data to numpy arrays, and save whole result to pkl format.

    The data are stored as dictionary. Each feature's data can be accessed by calling self.data['feat_name'].

    !Warning!
    For reader.py's purposes, the data dictionary contains extra feature called 'criterion012345'. This feature stores
    root criterion used for extracting the data from original tree. As an end user you should always access the data
    from reader.py instance which automatically returns data without this feature.
    """

    def __init__(self, params={}):
        self.verbose = params.get('verbose')
        self.state = 0
        self.file = {}
        self.data = {}
        self.filename = ''
        self.directory = ''
        self.criterion = ''
        self.logger = logging.getLogger()
    def open(self, f):
        """ Open specified root file.
        @param self:
        @param f: The root file which is to be open.
        @return:
        """

        if self.verbose is not None:
            self.logger.info('converter.open():')
            self.logger.info('\t|-> opening file \'' + f + '\'')

        self.file = uproot.open(f)
        self.filename = f.rsplit('/', 1)[1][:-5]
        self.logger.info("processing file "+ str(self.filename))
        self.directory = f.rsplit('/', 1)[0]
        self.state = 1

        if self.verbose is not None:
            self.logger.info('\t--------- open done ----------')

    @staticmethod
    def to_code(string):
        """ Convert ROOT query to python language.
        @param string: The string which is to be converted to code_jm.
        @return: The converted string, The branches needed for successful query.
        """

        expressions = re.split(',|;', string)
        req_branches = []

        operators = []
        for c in string:
            if c == ',' or c == ';':
                operators.append(c)

        for i, expression in enumerate(expressions):
            if expression.find('==') != -1:
                branch = re.split('==', expression)
                req_branches.append(branch[0])
                code = re.sub('[a-z,A-Z,0-9,_]+', lambda m: 'dc[\'%s\']' % m.group(0), branch[0])
                expressions[i] = code + '==' + branch[1]
            elif expression.find('!=') != -1:
                branch = re.split('!=', expression)
                req_branches.append(branch[0])
                code = re.sub('[a-z,A-Z,0-9,_]+', lambda m: 'dc[\'%s\']' % m.group(0), branch[0])
                expressions[i] = code + '!=' + branch[1]
            elif expression.find('<') != -1:
                branch = re.split('<', expression)
                req_branches.append(branch[0])
                code = re.sub('[a-z,A-Z,0-9,_]+', lambda m: 'dc[\'%s\']' % m.group(0), branch[0])
                expressions[i] = code + '<' + branch[1]
            elif expression.find('>') != -1:
                branch = re.split('>', expression)
                req_branches.append(branch[0])
                code = re.sub('[a-z,A-Z,0-9,_]+', lambda m: 'dc[\'%s\']' % m.group(0), branch[0])
                expressions[i] = code + '>' + branch[1]
            elif expression.find('<=') != -1:
                branch = re.split('<=', expression)
                req_branches.append(branch[0])
                code = re.sub('[a-z,A-Z,0-9,_]+', lambda m: 'dc[\'%s\']' % m.group(0), branch[0])
                expressions[i] = code + '<=' + branch[1]
            elif expression.find('>=') != -1:
                branch = re.split('>=', expression)
                req_branches.append(branch[0])
                code = re.sub('[a-z,A-Z,0-9,_]+', lambda m: 'dc[\'%s\']' % m.group(0), branch[0])
                expressions[i] = code + '>=' + branch[1]
            else:
                print('Unsupported comparison!')

        code = ''
        if len(expressions) > 1:
            for i, expression in enumerate(expressions):
                if i < len(operators):
                    code += expression + operators[i]
                else:
                    code += expression
        else:
            code = expressions[0]

        code = '(' + code + ')'
        code = code.replace(',', ') and (')
        code = code.replace(';', ' or ')

        return code, req_branches

    def scan(self, tree_name, branches='*', criterion=None):
        """ Scan the file.
        @param self:
        @param tree_name: The name of tree used in query.
        @param branches: Branches on which the request is made.
        @param criterion: Query criterion.
        @return:
        """

        if self.state < 1:
            warnings.warn('Warning: File need to be opened first! Use method open()')
            return

        if self.verbose is not None:
            self.logger.info('converter.scan():')

        dc = {}
        separator = ':'
        self.criterion = criterion  # for passing to saved file

        # Handle ROOT notation
        if branches == '*':
            branches = self.file[tree_name].keys()
            branches = list(dict.fromkeys(branches))
            #print(branches)
        else:
            branches = re.split(separator, branches)

        # Handle existence of file
        for i, branch in enumerate(branches):
            if not isinstance(branch, str):
                branches[i] = branch.decode('utf-8')
                # convert to utf-8 (remove 'b prefix)
        if path.exists(self.directory + '/' + self.filename + '.pkl'):
            rd = reader.Reader()
            rd.open(self.directory + '/' + self.filename + ".pkl")

            saved_query = rd.get_query()
            our_query = separator.join(branches)

            if criterion:
                saved_criterion = rd.get_criterion()
                if saved_query == our_query and saved_criterion == criterion:
                    raise FileExistsError
            elif saved_query == our_query:
                raise FileExistsError

        # Get data of branches
        for i, branch in enumerate(branches):
            if self.verbose is not None:
                self.logger.info('\t|-> reading branch \'' + branches[i] + '\'')

            dc[branches[i]] = self.file[tree_name].array(branches[i])


        # Apply criterion if it exists
        if criterion:
            criterion, req_branches = self.to_code(criterion)
            if self.verbose is not None:
                self.logger.info('\t------------------------------')
                self.logger.info('\t|-> applying criterion \'' + criterion + '\'')
            for branch in req_branches:
                if branch not in dc:  # check if branch is missing in dictionary
                    if self.verbose is not None:
                        self.logger.info('\t\t|-> reading branch \'' + branch + '\'')
                    dc[branch] = self.file[tree_name].array(branch)

            result = eval(criterion)
            for key in dc:
                dc[key] = dc[key][result]

        # Remove unwanted branches from dc (if they were added due to criterion req)
        dc = {key: dc[key] for key in branches}

        self.data = dc
        self.state = 2

        if self.verbose is not None:
            self.logger.info('\t--------- scan done ----------')

    def convert(self):
        """ Convert data to numpy arrays.
        @param self:
        @return:
        """

        if self.state < 2:
            warnings.warn('Warning: File need to be scanned first! Use method scan()')
            return

        if self.verbose is not None:
            self.logger.info('converter.convert():')
            self.logger.info('\t|-> converting data to numpy arrays')

        for key in self.data:
            self.data[key] = np.array(self.data[key])

        self.state = 3

        if self.verbose is not None:
            self.logger.info('\t-------- convert done --------')

    def save(self, folder, name):
        """ Save data in pkl format.
        @param self:
        @param folder: Name of the folder
        @param name: Name of the file
        @return:
        """

        if self.state < 3:
            warnings.warn('Warning: File need to be converted first! Use method convert()')
            return

        if self.verbose is not None:
            self.logger.info('converter.save():')
            self.logger.info('\t|-> saving data to pkl format')

        # Add criterion to dictionary
        self.data['criterion012345'] = self.criterion

        with open(folder + name + '.pkl', 'wb') as f:
            pickle.dump(self.data, f)
            f.close()

        self.state = 4

        if self.verbose is not None:
            self.logger.info('\t--------- save done ----------')

