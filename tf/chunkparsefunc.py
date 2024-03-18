#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2021 Leela Chess Authors
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
import tensorflow as tf

HISTORY = 2

def parse_function(planes, probs):
    """
    Convert unpacked record batches to tensors for tensorflow training
    """
    planes = tf.io.decode_raw(planes, tf.float32)
    probs = tf.io.decode_raw(probs, tf.float32)

    planes = tf.reshape(planes, (-1, 18*HISTORY, 8, 8))
    probs = tf.reshape(probs, (-1, 1792))

    return (planes, probs)
