
// Linear normalization
function lnorm(val, min, max) {
    return (val - min) / (max - min);
}

// Function to linearly interpolate between a0 and a1
// Weight w should be in the range [0.0, 1.0]
function lerp(a0, a1, w) {
    return (1.0 - w)*a0 + w*a1;
}

function parseHexColor(colorStr) {
    return {
        r: parseInt(colorStr.substr(1, 2), 16),
        g: parseInt(colorStr.substr(3, 2), 16),
        b: parseInt(colorStr.substr(5, 2), 16)
    };
}

function mix(color1, color2, k) {
    return {
        r: lerp(color1.r, color2.r, k),
        g: lerp(color1.g, color2.g, k),
        b: lerp(color1.b, color2.b, k)
    };
}

function pad(str, ch, width) {
    while (str.length < width) str = ch + str;
    return str;
}

function toHex(color) {
    let r = pad(Math.floor(color.r).toString(16), '0', 2);
    let g = pad(Math.floor(color.g).toString(16), '0', 2);
    let b = pad(Math.floor(color.b).toString(16), '0', 2);
    return `#${r}${g}${b}`;
}

// Color lookup table
class LUT {
    constructor() {
        this.colors = []
        this.maxValue = null
        this.minValue = null
        this.customDraw = false

        this.canvas = document.createElement('canvas');
        this.width = this.canvas.width = 32;
        this.height = this.canvas.height = 256;
        document.body.appendChild(this.canvas);
    }

    addStop(value, color) {
        this.maxValue = Math.max(value, this.maxValue);
        this.minValue = Math.min(value, this.minValue);
        this.colors.push({ value: value, color: color });
        this.colors.sort((a, b) => a.value - b.value);
    }

    // TODO: fix me?
    getColor(value) {
        value = lerp(this.minValue, this.maxValue, value);
        if (value > this.maxValue) return this.colors[this.colors.length - 1].color;
        if (value < this.minValue) return this.colors[0].color;

        let i = 1;
        // console.log(this.colors[1]);
        for (i = 1; i < this.colors.length; i++) {
            // console.log(i, this.colors[i].value, value);
            if (this.colors[i].value >= value) break;
        }
        // blend color i and i-1
        let k = lnorm(value, this.colors[i-1].value, this.colors[i].value);
        let color = mix(parseHexColor(this.colors[i-1].color), parseHexColor(this.colors[i].color), k);
        return toHex(color);
    }

    draw() {
        let ctx = this.canvas.getContext('2d');
        let grd = ctx.createLinearGradient(0, 0, 0, this.height);
        for (let stop of this.colors) {
            let k = lnorm(stop.value, this.minValue, this.maxValue);
            grd.addColorStop(k, stop.color);
        }
        ctx.fillStyle = grd;
        ctx.fillRect(0, 0, this.width, this.height);
    }

    drawDiscrete(n_steps) {
        let ctx = this.canvas.getContext('2d');
        let start = 0;
        let end = 0;
        // let n = this.colors.length;
        let step = this.height / n_steps;
        // for (let stop of this.colors) {
        for (let i = 0; i < n_steps; i++) {
            // let stop = this.stops[n];
            // let k = lnorm(stop.value, this.minValue, this.maxValue);
            let color = this.getColor((i+0.5) / n_steps);
            // let r = Math.floor(256 * ((i+0.5) / n_steps));
            // let color = `rgb(${r},${r},${r})`;
            end = start + step;
            ctx.fillStyle = color;
            ctx.fillRect(0, Math.floor(start), this.width, Math.floor(end));
            start = end;
        }
    }

    drawCustom(n_steps, draw_fn) {
        let ctx = this.canvas.getContext('2d');
        draw_fn(ctx, n_steps);
    }
}

const INTERPOLATION_MODE_NEAREST = 2;
const INTERPOLATION_MODE_LINEAR = 1;
const INTERPOLATION_MODE_CUBIC = 0;

class ColorMapGL {
    constructor(gl, luts, dataSet) {
        this.gl = gl;
        this.luts = luts;
        this.dataSet = dataSet;
        this.colorScale = this.gl.createTexture();
        this.map = this.gl.createTexture();
        this.dataTex0 = gl.createTexture();
        this.dataTex1 = gl.createTexture();
        this.programInfo = null;
        this.buffers = null;
        this.data0 = {
            values: [],
            width: 0,
            height: 0
        };
        this.data1 = {
            values: [],
            width: 0,
            height: 0
        };

        this.options = {};
        this.options.data0 = 'none';
        this.options.data1 = 'none';
        this.options.lut = 'rainbow';
        this.options.alpha = 0.5;
        this.options.discrete = true;
        this.options.n_steps = 16;
        this.options.noise = 0.035;
        this.options.interpolation = INTERPOLATION_MODE_LINEAR;
        this.options.frame = 0;
        this.options.frames_per_second = 0.5;

        const vsSource = `
        attribute vec4 aVertexPosition;
        attribute vec2 aTextureCoord;
        uniform mat4 uModelViewMatrix;
        uniform mat4 uProjectionMatrix;
        varying vec2 vTextureCoord;

        void main() {
            gl_Position = aVertexPosition;
            vTextureCoord = aTextureCoord;
        }
        `;

        const fsSource = `
        precision mediump float;
        varying vec2 vTextureCoord;
        uniform sampler2D uData0Sampler;
        uniform sampler2D uData1Sampler;
        uniform sampler2D uColorScaleSampler;
        uniform sampler2D uMapSampler;
        uniform float k_noise;
        uniform float k_alpha;
        uniform float frame;
        uniform vec2 texSize0;
        uniform vec2 texSize1;
        uniform int interpolationMode;

        float rand(vec2 co) {
            return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
        }

        vec2 circ(float theta) {
            return vec2(cos(theta), sin(theta));
        }

        vec2 generateGaussianNoise(float sigma, vec2 co) {
            float two_pi = 2.0*3.14159265358979323846;
            float epsilon = 0.0000000000001;
            vec2 u = vec2(rand(co), rand(co));
            vec2 z = sqrt(-2.0 * log(u)) * circ(two_pi * u.y);
            return z * sigma;
        }

        float bezier(float t, vec4 ww) {
            float s = 1.0 - t;
            vec4 ss = vec4(s*s*s, s*s, s, 1.0);
            vec4 tt = vec4(1.0, t, t*t, t*t*t);
            vec4 cc = vec4(1.0, 3.0, 3.0, 1.0);
            return dot(ww*ss*tt, cc);
        }

        float tang(float t, float k) {
            float pi = 3.141592653582;
            return tan(k*pi*(t - 0.5)) / (2.0*tan(0.5*k*pi));
        }

        vec4 cubic(float v)
        {
            vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
            vec4 s = n * n * n;
            float x = s.x;
            float y = s.y - 4.0 * s.x;
            float z = s.z - 4.0 * s.y + 6.0 * s.x;
            float w = 6.0 - x - y - z;
            return vec4(x, y, z, w);
        }

        vec4 bicubic(sampler2D texture, vec2 texcoord, vec2 texSize)
        {
            // vec2 texSize = textureSize(texture, 0);
            // vec2 texSize = vec2(8.0, 8.0);
            vec2 invTexSize = 1.0 / texSize;

            texcoord = texcoord * texSize - 0.5;

            vec2 fxy = fract(texcoord);
            texcoord -= fxy;

            vec4 xcubic = cubic(fxy.x);
            vec4 ycubic = cubic(fxy.y);

            vec4 c = texcoord.xxyy + vec2(-0.5, 1.5).xyxy;
            vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
            vec4 offset = c + vec4(xcubic.yw, ycubic.yw) / s;

            offset *= invTexSize.xxyy;
            vec4 sample0 = texture2D(texture, offset.xz);
            vec4 sample1 = texture2D(texture, offset.yz);
            vec4 sample2 = texture2D(texture, offset.xw);
            vec4 sample3 = texture2D(texture, offset.yw);

            float sx = s.x / (s.x + s.y);
            float sy = s.z / (s.z + s.w);

            return mix(
                mix(sample3, sample2, sx),
                mix(sample1, sample0, sx),
                sy
            );
        }

        float luma(vec4 a) {
            return dot(a.rgb, vec3(1.0)) / 3.0;
        }

        vec4 multiply(vec4 a, vec4 b) {
            return a * b;
        }

        vec4 screen(vec4 a, vec4 b) {
            return vec4(1.0) - (vec4(1.0) - a) * (vec4(1.0) - b);
        }

        vec4 overlay(vec4 a, vec4 b) {
            if (luma(a) < 0.5) return vec4(2.0) * a * b;
            else return vec4(1.0) - vec4(2.0) * (vec4(1.0) - a) * (vec4(1.0) - b);
        }

        void main() {
            vec4 samp0, samp1, samp;
            if (interpolationMode == 0) {
                samp0 = bicubic(uData0Sampler, vTextureCoord, texSize0);
                samp1 = bicubic(uData1Sampler, vTextureCoord, texSize1);
            } else {
                samp0 = texture2D(uData0Sampler, vTextureCoord);
                samp1 = texture2D(uData1Sampler, vTextureCoord);
            }
            samp = mix(samp0, samp1, frame);
            // samp = samp0;

            float noise_uniform = rand(vTextureCoord);
            float noise = k_noise * tang(noise_uniform, sqrt(k_alpha));
            // float noise = k_noise * (noise_uniform - 0.5);

            float val = clamp(samp.r + noise, 0.0, 255.0 / 256.0);
            vec4 mapped_frag = texture2D(uColorScaleSampler, vec2(0.0, val));
            // vec4 bg_frag = texture2D(uMapSampler, vTextureCoord);
            // gl_FragColor = mix(mapped_frag, bg_frag, k_alpha);
            // gl_FragColor = (mapped_frag + bg_frag) / (vec4(1.0) + bg_frag);
            // gl_FragColor = overlay(mapped_frag, bg_frag);
            gl_FragColor = mapped_frag;
        }
        `;

        const shaderProgram = this.initShaderProgram(vsSource, fsSource);
        this.programInfo = {
            program: shaderProgram,
            attribLocations: {
                vertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
                textureCoord: gl.getAttribLocation(shaderProgram, 'aTextureCoord'),
                data: gl.getAttribLocation(shaderProgram, 'aData'),
            },
            uniformLocations: {
                projectionMatrix: gl.getUniformLocation(shaderProgram, 'uProjectionMatrix'),
                modelViewMatrix: gl.getUniformLocation(shaderProgram, 'uModelViewMatrix'),
                uData0Sampler: gl.getUniformLocation(shaderProgram, 'uData0Sampler', 0),
                uData1Sampler: gl.getUniformLocation(shaderProgram, 'uData1Sampler', 0),
                uColorScaleSampler: gl.getUniformLocation(shaderProgram, 'uColorScaleSampler', 1),
                uMapSampler: gl.getUniformLocation(shaderProgram, 'uMapSampler', 2),
                k_noise: gl.getUniformLocation(shaderProgram, 'k_noise'),
                k_alpha: gl.getUniformLocation(shaderProgram, 'k_alpha'),
                frame: gl.getUniformLocation(shaderProgram, 'frame'),
                texSize0: gl.getUniformLocation(shaderProgram, 'texSize0'),
                texSize1: gl.getUniformLocation(shaderProgram, 'texSize1'),
                interpolationMode: gl.getUniformLocation(shaderProgram, 'interpolationMode'),
            },
        };
        gl.useProgram(shaderProgram);

        this.buffers = this.initBuffers();

        this.gl.bindTexture(this.gl.TEXTURE_2D, this.dataTex0);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
        
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.dataTex1);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);

        this.gl.bindTexture(this.gl.TEXTURE_2D, this.map);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);

        this.setColorScale();
        this.setData();
    }

    setBackground(img) {
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.map);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, img);
    }

    // x y on range [-1,1]
    f(x, y) {
        // return (Math.cos(Math.PI * (x + y)) + Math.cos(Math.PI * (x - y)) + 2) / 4;
        x *= 5;
        y *= 5;
        // return 1 - (1 + Math.cos(x*x+y)) / 2;
        return Math.min(Math.max(0, Math.sin(x*x + y*y) * x / Math.PI), 0.999);
    }

    createData() {
        let data = {
            width: 32,
            height: 32,
            values: []
        };
        for (let y = 0; y < data.height; y++) {
            for (let x = 0; x < data.width; x++) {
                let i = y * data.width + x;
                let s = (x - data.width/2);
                let t = (y - data.height/2);
                // let val = s*s + t*t;
                // val = (val + 2) / (val + 1) - 1;
                // val = (Math.cos(s+t) + Math.cos(s-t) + 2) / 4;
                // data.values[i] = Math.max(0, Math.min(val, 255/256));
                // data.values[i] = Math.floor(1000*Math.random())/1000;
                let f = this.f(2 * (x+0.5) / data.width - 1, 2 * (y+0.5) / data.height - 1);
                data.values[i] = Math.floor(1000 * f) / 1000;
                // data.values[i] = (x + y) / 14;
            }
        }
        console.log(JSON.stringify(data));
    }

    setData() {
        if (!this.dataSet.hasOwnProperty(this.options.data0)) {
            throw new Error(`${this.options.data} is not a defined dataSet.`);
            return;
        }
        if (!this.dataSet.hasOwnProperty(this.options.data1)) {
            throw new Error(`${this.options.data} is not a defined dataSet.`);
            return;
        }
        // this.data0 = JSON.parse(this.dataSet[this.options.data0]);
        // this.data1 = JSON.parse(this.dataSet[this.options.data1]);
        this.data0 = this.dataSet[this.options.data0];
        this.data1 = this.dataSet[this.options.data1];
        this.generateData();
    }

    generateData() {
        let n = this.data0.values.length;
        let pixels = new Uint8Array(2 * n);
        for (let i = 0; i < n; i++) {
            let j = 2 * i;
            let val = Math.floor(256 * this.data0.values[i]);
            pixels[j] = val;
            pixels[j+1] = 0xff;
        }
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.dataTex0);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.LUMINANCE_ALPHA, this.data0.width, this.data0.height, 0, this.gl.LUMINANCE_ALPHA, this.gl.UNSIGNED_BYTE, pixels);

        n = this.data1.values.length;
        pixels = new Uint8Array(2 * n);
        for (let i = 0; i < n; i++) {
            let j = 2 * i;
            let val = Math.floor(256 * this.data1.values[i]);
            pixels[j] = val;
            pixels[j+1] = 0xff;
        }
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.dataTex1);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.LUMINANCE_ALPHA, this.data1.width, this.data1.height, 0, this.gl.LUMINANCE_ALPHA, this.gl.UNSIGNED_BYTE, pixels);
    }

    setColorScale() {
        let lut = this.luts[this.options.lut];
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.colorScale);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, lut.canvas);
    }

    // Creates a shader of the given type, uploads the source and compiles it.
    loadShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            // console.error('An error occurred compiling the shaders: ' + this.gl.getShaderInfoLog(shader));
            throw new Error('An error occurred compiling the shaders: ' + this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    // Initialize a shader program, so WebGL knows how to draw our data
    initShaderProgram(vsSource, fsSource) {
        const vertexShader = this.loadShader(this.gl.VERTEX_SHADER, vsSource);
        const fragmentShader = this.loadShader(this.gl.FRAGMENT_SHADER, fsSource);
        const shaderProgram = this.gl.createProgram();
        this.gl.attachShader(shaderProgram, vertexShader);
        this.gl.attachShader(shaderProgram, fragmentShader);
        this.gl.linkProgram(shaderProgram);

        if (!this.gl.getProgramParameter(shaderProgram, this.gl.LINK_STATUS)) {
            // console.error('Unable to initialize the shader program: ' + this.gl.getProgramInfoLog(shaderProgram));
            throw new Error('Unable to initialize the shader program: ' + this.gl.getProgramInfoLog(shaderProgram));
            return null;
        }

        return shaderProgram;
    }

    initBuffers() {
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = [ 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0 ];
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const textureCoordBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, textureCoordBuffer);
        const textureCoordinates = [ 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0 ];
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(textureCoordinates), this.gl.STATIC_DRAW);

        return {
            position: positionBuffer,
            texture: textureCoordBuffer
        };
    }

    draw() {
        let gl = this.gl;
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.uniform1f(this.programInfo.uniformLocations.k_noise, this.options.noise);
        gl.uniform1f(this.programInfo.uniformLocations.k_alpha, this.options.alpha);
        gl.uniform2f(this.programInfo.uniformLocations.texSize0, this.data0.width, this.data0.height);
        gl.uniform2f(this.programInfo.uniformLocations.texSize1, this.data1.width, this.data1.height);
        gl.uniform1f(this.programInfo.uniformLocations.frame, this.options.frame);
        gl.uniform1i(this.programInfo.uniformLocations.interpolationMode, this.options.interpolation);

        this.gl.bindTexture(this.gl.TEXTURE_2D, this.dataTex0);
        if (this.options.interpolation == INTERPOLATION_MODE_NEAREST) {
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
        } else {
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        }

        this.gl.bindTexture(this.gl.TEXTURE_2D, this.dataTex1);
        if (this.options.interpolation == INTERPOLATION_MODE_NEAREST) {
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
        } else {
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        }

        // Data texture 0
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.dataTex0);
        gl.uniform1i(this.programInfo.uniformLocations.uData0Sampler, 0);
        
        // Data texture 1
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.dataTex1);
        gl.uniform1i(this.programInfo.uniformLocations.uData1Sampler, 1);

        // Color scale texture
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, this.colorScale);
        gl.uniform1i(this.programInfo.uniformLocations.uColorScaleSampler, 2);

        // Map texture
        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, this.map);
        gl.uniform1i(this.programInfo.uniformLocations.uMapSampler, 3);

        // Vertex position buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
        gl.vertexAttribPointer(this.programInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.programInfo.attribLocations.vertexPosition);

        // Texture coordinate buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.texture);
        gl.vertexAttribPointer(this.programInfo.attribLocations.textureCoord, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.programInfo.attribLocations.textureCoord);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
}

window.addEventListener('load', (evt) => {
    let luts = {};

    luts.greyscale = new LUT();
    luts.greyscale.addStop(0, '#ffffff');
    luts.greyscale.addStop(1, '#000000');
    luts.greyscale.draw();

    luts.blue = new LUT();
    luts.blue.addStop(0, '#f0f9e8');
    luts.blue.addStop(1, '#bae4bc');
    luts.blue.addStop(2, '#7bccc4');
    luts.blue.addStop(3, '#43a2ca');
    luts.blue.addStop(4, '#0868ac');
    luts.blue.draw();

    luts.rainbow = new LUT();
    luts.rainbow.addStop(0, '#741A14');
    luts.rainbow.addStop(1, '#CC362C');
    luts.rainbow.addStop(2, '#EC6C33');
    luts.rainbow.addStop(3, '#F7BD33');
    luts.rainbow.addStop(4, '#E9F636');
    luts.rainbow.addStop(5, '#A6F57F');
    luts.rainbow.addStop(6, '#7CF7DC');
    luts.rainbow.addStop(7, '#50C0FB');
    luts.rainbow.addStop(8, '#0072FB');
    luts.rainbow.addStop(9, '#003EDC');
    luts.rainbow.addStop(10, '#00248D');
    luts.rainbow.draw();

    function topoDraw(ctx, n_steps) {
        let h = ctx.canvas.height / n_steps;
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.fillStyle = "black";
        for (let i = 0; i < n_steps; i++)
            ctx.fillRect(0, i * h, ctx.canvas.width, 1);
    }

    luts.topo = new LUT();
    luts.topo.customDraw = true;
    luts.topo.drawCustom(32, topoDraw);

    let dataSet = {
        'none': {"values":[0],"width":1,"height":1}
    };

    let canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.id = "colormap";
    document.body.appendChild(canvas);

    let gl = canvas.getContext('webgl');
    let colorMap = new ColorMapGL(gl, luts, dataSet);

    // Load background map image
    // var map = new Image();
    // map.src = "map.png";
    // map.addEventListener('load', () => {
        // colorMap.setBackground(map);
        colorMap.draw();
    // });

    let gui = new dat.GUI();
    // gui.add(colorMap, 'createData');
    // let dat_data0 = gui.add(colorMap.options, 'data0', Object.keys(dataSet)).onChange((val) => {
    //     colorMap.setData();
    //     colorMap.draw();
    // });
    // let dat_data1 = gui.add(colorMap.options, 'data1', Object.keys(dataSet)).onChange((val) => {
    //     colorMap.setData();
    //     colorMap.draw();
    // });

    function flatten(arr) {
        return arr.reduce((prev, curr) => {
            return prev.concat(curr);
        }, []);
    }

    function load_data(timestamp) {
        let url = `01/gridData_${timestamp}.json`;
        return window.fetch(url, { method: 'GET' }).then((response) => {
            if (response.ok) {
                return response.json().then((json) => {
                    // console.log(json);
                    let values = flatten(json['grid']);
                    let minVal = values.reduce((min, cur) => {return cur < min ? cur : min}, values[0]);
                    let maxVal = values.reduce((max, cur) => {return cur > max ? cur : max}, values[0]);
                    colorMap.dataSet[timestamp] = {
                        'width': json['grid'][0].length,
                        'height': json['grid'].length,
                        'values': values.map(x => 0.999*lnorm(x, minVal, maxVal))
                    };
                    // callback(colorMap.dataSet[timestamp]);
                });
            }
        });
    }

    let delta = 300;
    let start = 1362117600;
    let end = 1362181200;
    // let end = start + 5*delta;

    let data_frames = [];
    for (t = start; t < end; t += delta) {
        data_frames.push(t.toString());
    }

    Promise.all(data_frames.map(load_data)).then(() => {
        console.log('Loaded all data sets');
        requestAnimationFrame(animate);
    });

    function redraw() {
        colorMap.setColorScale();
        colorMap.draw();
    }

    let frame_t = 0;
    let frame_a = 0;
    let frame_b = 1;
    let last_draw_t = 0;

    function animate(t) {
        let dt = t - last_draw_t;
        frame_t += dt * colorMap.options.frames_per_second / 1000;
        if (frame_t >= data_frames.length) {
            frame_t -= data_frames.length;
        }
        frame_a = Math.floor(frame_t);
        frame_b = frame_a + 1;
        if (frame_b >= data_frames.length) {
            frame_b -= data_frames.length;
        }
        colorMap.options.data0 = data_frames[frame_a];
        colorMap.options.data1 = data_frames[frame_b];
        colorMap.options.frame = frame_t - frame_a;
        colorMap.setData();
        colorMap.draw();

        last_draw_t = t;
        requestAnimationFrame(animate);
    }

    function redrawLUTs() {
        console.log('redrawing LUTs');
        for (let lut in luts) {
            if (luts.hasOwnProperty(lut)) {
                if (luts[lut].customDraw) luts[lut].drawCustom(colorMap.options.n_steps, topoDraw);
                else if (colorMap.options.discrete) luts[lut].drawDiscrete(colorMap.options.n_steps);
                else luts[lut].draw()
            }
        }
        redraw();
    }

    gui.add(colorMap.options, 'lut', Object.keys(luts)).onChange(redraw);
    gui.add(colorMap.options, 'n_steps').min(1).max(64).step(1).onChange(redrawLUTs);
    gui.add(colorMap.options, 'discrete').onChange(redrawLUTs);
    gui.add(colorMap.options, 'noise').min(0).max(0.25).onChange(colorMap.draw.bind(colorMap));
    gui.add(colorMap.options, 'alpha').min(0).max(1).onChange(colorMap.draw.bind(colorMap));
    gui.add(colorMap.options, 'interpolation', {
        'nearest': INTERPOLATION_MODE_NEAREST,
        'linear': INTERPOLATION_MODE_LINEAR,
        'cubic': INTERPOLATION_MODE_CUBIC
    }).onChange(colorMap.draw.bind(colorMap));
    // gui.add(colorMap.options, 'frame').min(0).max(1).step(0.01).onChange(colorMap.draw.bind(colorMap)).listen();
});
