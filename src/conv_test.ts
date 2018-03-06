/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

import { test } from "../tools/tester";
import { range, Tensor } from "./api";
import { assertAllClose } from "./tensor_util";
import { DType, ImageFormat, Padding } from "./types";

// For easy sanity checking, the test cases were ported from
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/1f441c191f9a6d8f27b32b1c19c55f76aaf9e387/tensorflow/compiler/tests/conv2d_test.py#L79-L177
// Dilation tests skipped entirely for now.

interface ConvExample {
  inputShape: [number, number, number, number]; // NHWC
  filterShape: [number, number, number, number]; // H W InChans OutChans
  strides: [number, number]; // [column, row ]
  padding: Padding;
  expected: number[];
}

test(async function testConv2D1x1Filter() {
  verify({
    inputShape: [1, 2, 3, 3],
    filterShape: [1, 1, 3, 3],
    strides: [1, 1],
    padding: "valid",
    expected: [
      30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0,
      171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ],
  });
});

/* TODO DL Fails here.
test(async function testConv2DEmpty() {
  verify({
    inputShape: [0, 2, 3, 3],
    filterShape: [1, 1, 3, 3],
    strides: [1, 1],
    padding: "valid",
    expected: [],
  });
});
*/

test(async function testConv2D2x2Filter() {
  verify({
    inputShape: [1, 2, 3, 3],
    filterShape: [2, 2, 3, 3],
    strides: [1, 1],
    padding: "valid",
    expected: [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0],
  });
});

test(async function testConv2D1x2Filter() {
  verify({
    inputShape: [1, 2, 3, 3],
    filterShape: [1, 2, 3, 3],
    strides: [1, 1],
    padding: "valid",
    expected: [ 231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0,
      840.0, 843.0, 936.0, 1029.0 ]
  });
});

test(async function testConv2D2x2FilterStride2() {
  verify({
    inputShape: [1, 2, 3, 3],
    filterShape: [2, 2, 3, 3],
    strides: [2, 2],
    padding: "valid",
    expected: [2271.0, 2367.0, 2463.0],
  });
});

test(async function testConv2D2x2FilterStride2Same() {
  verify({
    inputShape: [1, 2, 3, 3],
    filterShape: [2, 2, 3, 3],
    strides: [2, 2],
    padding: "same",
    expected: [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
  });
});

function verify(ex: ConvExample) {
  const formats: ImageFormat[] = ["NHWC"];  // TODO test NCHW.
  for (const format of formats) {
    const actual = setupValues(format, "float32", ex);
    assertAllClose(actual.dataSync(), ex.expected);
  }
}

function setupValues(format: ImageFormat, dtype: DType,
                     ex: ConvExample): Tensor {
  let input = range(1, prod(ex.inputShape) + 1)
    .cast(dtype).reshape(ex.inputShape);
  if (format === "NCHW") input = NCHWToNHWC(input);
  const filter = range(1, prod(ex.filterShape) + 1)
    .cast(dtype).reshape(ex.filterShape);
  let r = input.conv2d(filter, {
    strides: ex.strides,
    padding: ex.padding,
    format,
  });
  if (format === "NCHW") r = NCHWToNHWC(r);
  return r;
}

// Ideally this would be an api.ts function.
function prod(array: number[]): number {
  return array.reduce((a, b) => a * b);
}

function NCHWToNHWC(t: Tensor): Tensor {
  return t.transpose([0, 3, 2, 1]);
}
