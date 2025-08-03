class Vector2 {
    constructor(x = 0, y = 0) {
        this.x = x;
        this.y = y;
    }

    /**
     * @param {Vector2} v
     * @param {number} s
     */
    addEqual(v, s = 1.0) {
        this.x += v.x * s;
        this.y += v.y * s;
    }

    /**
     * @param {Vector2} v
     */
    add(v, s=1.0) {
        return new Vector2(this.x + s*v.x, this.y + s*v.y);
    }

    /**
     * @param {{ x: number; y: number; }} v
     */
    subtract(v) {
        return new Vector2(this.x - v.x, this.y - v.y);
    }
    
    /**
     * @param {{ x: number; y: number; }} v
     * @param {number} s
     */
    subtractEqual(v, s = 1.0) {
        this.x -= s * v.x;
        this.y -= s * v.y;
    }
    
    /**
     * @param {number} s
     */
    timesEqual(s) {
        this.x *= s;
        this.y *= s;
    }
    
    /**
     * @param {number} s
     */
    times(s) {
        return new Vector2(this.x * s, this.y * s);
    }

    
    /**
     * @param {Vector2} v
     */
    dot(v) {
        return this.x * v.x + this.y * v.y;
    }
    /**
     * @param {{ y: number; x: number; }} v
     */
    cross(v) {
        return this.x * v.y - this.y * v.x;
    }
    perpendicular() {
        return new Vector2(-this.y, this.x);
        //90 degrees counter-clockwise
    }
    length() {
        return Math.hypot(this.x, this.y);
        //hypotenuse, sqrt(x^2+y^2)
    }
    normalise() {
        let len = this.length();
        if (len == 0) {
            return new Vector2(0, 0);
        }
        return this.times(1 / len).clone();
    }
    normaliseEqual() {
        let len = this.length();
        if (len == 0) {
            this.x=0;
            this.y=0;
        }
        this.timesEqual(1 / len);
    }
    rotate(angle) {//angle in radians, anticlockwise
        let cosA = Math.cos(angle);
        let sinA = Math.sin(angle);
        return new Vector2(
            this.x * cosA - this.y * sinA,
            this.x * sinA + this.y * cosA
        );
    }
    clone() {
        return new Vector2(this.x, this.y);
    }
}


// Vector math in 2d space. 
// It makes more sense to just say something=VectorMath2.subtract(v1,v2) than something=v1.subtract(v2)
// But since there is no such thing like operator overloading in JS like what is usually done in C++ or C#, so I have to use this way.

// "const" constructs A namespace
const VectorMath2 = {
    add: (
        /** @type {Vector2} */ 
        v1, 
        /** @type {Vector2} */ 
        v2
    ) =>
        new Vector2(v1.x + v2.x, v1.y + v2.y),
    subtract: (
        /** @type {Vector2} */ 
        v1, 
        /** @type {Vector2} */ 
        v2
    ) => // => is a shorthand for function
        new Vector2(v1.x - v2.x, v1.y - v2.y),
    times: (
        /** @type {Vector2} */ 
        v, 
        /** @type {number} */ 
        s
    ) =>
        new Vector2(v.x * s, v.y * s),
    dot: (
        /** @type {Vector2} */
        v1, 
        /** @type {Vector2} */ 
        v2
    ) =>
        v1.x * v2.x + v1.y * v2.y,
    cross: (
        /** @type {Vector2} */ 
        v1, 
        /** @type {Vector2} */ 
        v2
    ) =>
        v1.x * v2.y - v1.y * v2.x,
    distance: (
        /** @type {Vector2} */ 
        v1, 
        /** @type {Vector2} */ 
        v2
    ) =>
        Math.hypot(v1.x-v2.x, v1.y-v2.y),
    zero: () =>
        new Vector2(0, 0),
};

export {Vector2, VectorMath2};
