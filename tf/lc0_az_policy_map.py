#!/usr/bin/env python3
import sys
import numpy as np
from policy_index import policy_index

columns = 'abcdefgh'
rows = '12345678'
promotions = 'rbq'  # N is encoded as normal move

col_index = {columns[i]: i for i in range(len(columns))}
row_index = {rows[i]: i for i in range(len(rows))}


def index_to_position(x):
    return columns[x[0]] + rows[x[1]]


def position_to_index(p):
    return col_index[p[0]], row_index[p[1]]


def valid_index(i):
    if i[0] > 7 or i[0] < 0:
        return False
    if i[1] > 7 or i[1] < 0:
        return False
    return True


def queen_move(start, direction, steps):
    i = position_to_index(start)
    dir_vectors = {
        'N': (0, 1),
        'NE': (1, 1),
        'E': (1, 0),
        'SE': (1, -1),
        'S': (0, -1),
        'SW': (-1, -1),
        'W': (-1, 0),
        'NW': (-1, 1)
    }
    v = dir_vectors[direction]
    i = i[0] + v[0] * steps, i[1] + v[1] * steps
    if not valid_index(i):
        return None
    return index_to_position(i)


def knight_move(start, direction, steps):
    i = position_to_index(start)
    dir_vectors = {
        'N': (1, 2),
        'NE': (2, 1),
        'E': (2, -1),
        'SE': (1, -2),
        'S': (-1, -2),
        'SW': (-2, -1),
        'W': (-2, 1),
        'NW': (-1, 2)
    }
    v = dir_vectors[direction]
    i = i[0] + v[0] * steps, i[1] + v[1] * steps
    if not valid_index(i):
        return None
    return index_to_position(i)

def is_even_possible(from_square: int, to_square: int): # 1792 total possible moves
    # Checks if the move from the from_square to the to_square is even possible given any piece
    from_rank, from_file = divmod(from_square, 8)
    to_rank, to_file = divmod(to_square, 8)
    
    # Check if the squares are the same
    if from_square == to_square:
        return False
    
    # Check if the move is horizontal or vertical
    if from_rank == to_rank or from_file == to_file:
        return True
    
    # Check if the move is diagonal
    if abs(from_rank - to_rank) == abs(from_file - to_file):
        return True
    
    # Check if the move is a knight move
    if abs(from_rank - to_rank) == 2 and abs(from_file - to_file) == 1:
        return True
    if abs(from_rank - to_rank) == 1 and abs(from_file - to_file) == 2:
        return True
    
    return False

def get_possible_moves_array_mapping():
    last_index = 0
    array_board = [-1] * 64*64
    for _from in range(64):
        for _to in range(64):
            if is_even_possible(_from, _to):
                array_board[64*_from+_to] = last_index
                last_index += 1
    return array_board
MOVE_TO_COMPRESSED_MOVE = get_possible_moves_array_mapping()

def make_map():
    az_to_lc0 = np.zeros((64 * 8 * 8, 1792), dtype=np.float32)
    # conv-position to 1792-vector
    for _from in range(64):
        for _to in range(64):
            if is_even_possible(_from, _to):
                i = MOVE_TO_COMPRESSED_MOVE[_from*64+_to]
                az_to_lc0[_from*64+_to][i] = 1
    return az_to_lc0



if __name__ == "__main__":
    # Generate policy map include file for lc0
    if len(sys.argv) != 2:
        raise ValueError(
            "Output filename is needed as a command line argument")

    az_to_lc0 = np.ravel(make_map('index'))
    header = \
"""/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2019 The LCZero Authors

 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace lczero {
"""
    line_length = 12
    with open(sys.argv[1], 'w') as f:
        f.write(header + '\n')
        f.write('const short kConvPolicyMap[] = {\\\n')
        for e, i in enumerate(az_to_lc0):
            if e % line_length == 0 and e > 0:
                f.write('\n')
            f.write(str(i).rjust(5))
            if e != len(az_to_lc0) - 1:
                f.write(',')
        f.write('};\n\n')
        f.write('}  // namespace lczero')
