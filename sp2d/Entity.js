import { Vector2, VectorMath2 } from "./Vector2.js";
import * as Draw from "./Draw.js";
//import * as Draw from "./Draw.js";

var eps = 1e-3;
// vel and angularVel will be set zero if is less this eps.

export class Rectangle {
    constructor(
        width,
        length,
        restitution = 0,
        friction=0,
        mass = 0,
        pos = VectorMath2.zero(),
        vel = VectorMath2.zero(),
        angle = 0,
        angularVel = 0,
        colour = "#007F00"
    ) {
        this.id = null;
        this.type = "Rectangle";

        this.length = length;
        this.width = width;

        this.restitution = restitution;
        this.friction=friction;

        this.mass = mass;
        if (!mass) {
            this.mass = length * width;
        }

        this.massInv = 1 / this.mass;
        this.inertia = (1 / 12) * this.mass * (this.length ** 2) + (1 / 12) * this.mass * (this.width ** 2);
        this.inertiaInv = 1 / this.inertia;

        this.pos = pos.clone();
        this.vel = vel.clone();
        this.angle = angle;
        this.angularVel = angularVel;

        this.vertices = this.getVertices();
        this.colour = colour; // String is passed by value. Strange.
    }

    setStatic() {
        this.mass = Infinity;
        this.inertia = Infinity;
        this.massInv = 0;
        this.inertiaInv = 0;
        this.vel = VectorMath2.zero();
    }

    /**
     * @param {number} dt
     * @param {Vector2} gravity
     */
    simulate(dt, gravity) {
        // console.log(gravity,dt);
        // console.log(this.width,this.length);
        this.vel.addEqual(gravity, dt);
        if (this.mass === Infinity) this.vel = VectorMath2.zero();
        this.pos.addEqual(this.vel, dt);
        this.angle += this.angularVel * dt;
        this.vertices = this.getVertices();

        // if (this.vel.length() < eps)
        //     this.vel = VectorMath2.zero();

        if (this.angularVel < eps)
            this.angularVel = 0;

        //this.vel.timesEqual();
    }

    draw()
    {
        Draw.drawRectangle(this.pos,this.angle,this.width,this.length,this.colour);
    }

    getVertices() {
        var halfWidth = this.width / 2;
        var halfLength = this.length / 2;
        var cosA = Math.cos(this.angle);
        var sinA = Math.sin(this.angle);

        return [
            new Vector2(
                this.pos.x + halfWidth * cosA - halfLength * sinA,
                this.pos.y + halfWidth * sinA + halfLength * cosA
            ), // Top-Right

            new Vector2(
                this.pos.x - halfWidth * cosA - halfLength * sinA,
                this.pos.y - halfWidth * sinA + halfLength * cosA
            ), // Top-Left

            new Vector2(
                this.pos.x - halfWidth * cosA + halfLength * sinA,
                this.pos.y - halfWidth * sinA - halfLength * cosA
            ), // Bottom-Left

            new Vector2(
                this.pos.x + halfWidth * cosA + halfLength * sinA,
                this.pos.y + halfWidth * sinA - halfLength * cosA
            )  // Bottom-Right
        ];
    }
}

export class Polygon {
    /**
     * 
     * @param {Vector2[]} vertices 
     */
    constructor(
        vertices, // vertices must be counterclockwise
        restitution = 0,
        friction=0,
        mass = 0,
        vel = VectorMath2.zero(),
        angle = 0,
        angularVel = 0,
        colour = "#00007F"
    ) {
        this.id = null;
        
        this.type = "Polygon";
        this.colour = colour;

        this.pos = Polygon.computeCentroid(vertices);
        //console.log(Polygon.computeCentroid(vertices));
        //console.log(this.pos);
        
        this.vertices = vertices.map(v => new Vector2(v.x, v.y));
        //shallow copy

        // Draw.drawPolygon(this.vertices,this.colour);
        // for(let i of this.vertices) {
        //     Draw.drawCircle(i,0.01, "#FF0000");
        // }

        this.localVertices = vertices.map(v => new Vector2(v.x-this.pos.x, v.y-this.pos.y));

        this.vel = vel.clone();
        this.angle = angle;
        this.angularVel = angularVel;
        this.restitution = restitution;
        this.friction=friction;

        if (!mass) {
            this.mass = Polygon.computeArea(vertices);;
        } else {
            this.mass = mass;
        }

        this.massInv = 1 / this.mass;
        this.inertia = Polygon.computeInertia(vertices, this.mass, this.pos);
        //console.log(this.pos);
        this.inertiaInv = 1 / this.inertia;
        //console.log(this.pos);

        // Draw.drawPolygon(this.vertices,this.colour);
        // Draw.drawCircle(this.pos,0.01,"#FF0000");
        // error;
    }

    simulate(dt, gravity) {

        // console.log(gravity,dt);
        // console.log(this.width,this.length);
        this.vel.addEqual(gravity, dt);
        if (this.mass === Infinity) this.vel = VectorMath2.zero();
        this.pos.addEqual(this.vel, dt);
        this.angle += this.angularVel * dt;
        this.vertices = this.getVertices();
        //this.localVertices = this.vertices.map(v => new Vector2(v.x-this.pos.x, v.y-this.pos.y));

        // if (this.vel.length() < eps)
        //     this.vel = VectorMath2.zero();

        if (this.angularVel < eps)
            this.angularVel = 0;
    }

    draw()
    {
        Draw.drawPolygon(this.vertices,this.colour);
    }

    static computeCentroid(vertices) {
        var sumv=VectorMath2.zero();
        for(let v of vertices)
        {
            sumv.addEqual(v);
        }
        // console.log("sumv:", sumv);
        // console.log("vertices.length:", vertices.length);
        // console.log("1 / vertices.length:", 1 / vertices.length);

        // console.log("res", 1 / vertices.length);
        var res=sumv.times(1/vertices.length).clone();
        //console.log(res);
        return res;

        //cetroid is the average of vertices.
    }

    static computeArea(vertices) {
        var area = 0;
        for (let i = 0; i < vertices.length; i++) {
            let j = (i + 1) % vertices.length;
            area += vertices[i].cross(vertices[j]);
            //console.log(vertices[i].cross(vertices[j]));
        }
        return Math.abs(area / 2);
    }

    static computeInertia(vertices, mass, centroid) {
        var inertia = 0;
        for (let i = 0; i < vertices.length; i++) {
            let j = (i + 1) % vertices.length;
            let p1 = vertices[i].subtract(centroid);
            let p2 = vertices[j].subtract(centroid);
            let cross = p1.cross(p2);
            inertia += (p1.dot(p1) + p1.dot(p2) + p2.dot(p2)) * cross;
        }
        return Math.abs((mass * inertia) / (6*Polygon.computeArea(vertices)));
    }

    getVertices() {
        var cosA = Math.cos(this.angle);
        var sinA = Math.sin(this.angle);
        return this.localVertices.map(v => new Vector2(
            this.pos.x + v.x * cosA - v.y * sinA,
            this.pos.y + v.x * sinA + v.y * cosA
        ));
    }
}

export class Circle {
    constructor(
        radius,
        restitution = 0,
        friction=0,
        mass = null,
        pos = VectorMath2.zero(),
        vel = VectorMath2.zero(),
        angle = 0,
        angularVel = 0,
        colour = "#FF0000"
    ) {
        this.id = null;
        this.radius = radius
        this.restitution = restitution;
        this.friction=friction;
        if (!mass)
            this.mass = Math.PI * radius ** 2;
        else
            this.mass=mass;
        this.massInv = 1 / this.mass;
        this.pos = pos.clone();
        this.vel = vel.clone();
        this.angle = 0;
        this.angularVel = 0;

        this.inertia = this.mass * (this.radius * this.radius) / 2;
        this.inertiaInv = 1 / this.inertia;
        
        this.colour = colour;
        this.type = "Circle";
    }

    setStatic() {
        this.mass = Infinity;
        this.massInv = 0;
        this.vel = VectorMath2.zero();
    }

    /**
     * @param {any} dt
     * @param {Vector2} gravity
     */
    simulate(dt, gravity) {
        //var kr = 0.01; //rolling friction canstant
        
        this.vel.addEqual(gravity.times(dt));
        if (this.mass === Infinity)
        {
            this.vel = VectorMath2.zero();
            //console.log(this.vel);
        }

        this.pos.addEqual(this.vel.times(dt));
        this.angle += this.angularVel * dt;
            

        //this.angularVel *= (1 - kr);

        // if (this.vel.length() < eps)
        //     this.vel = VectorMath2.zero();
    }

    draw()
    {
        Draw.drawCircle(this.pos,this.radius,this.colour);
    }
}

export class Particle {
    constructor(
        radius,
        restitution = 0,
        friction=0,
        mass = null,
        pos = VectorMath2.zero(),
        vel = VectorMath2.zero(),
        angle = 0,
        angularVel = 0,
        colour = "#FF0000"
    ) {
        this.id = null;
        this.radius = radius
        this.restitution = restitution;
        this.friction=friction;
        if (!mass)
            this.mass = Math.PI * radius ** 2;
        else
            this.mass=mass;
        this.massInv = 1 / this.mass;
        this.pos = pos.clone();
        this.vel = vel.clone();
        this.angle = 0;
        this.angularVel = 0;

        this.inertia = this.mass * (this.radius * this.radius) / 2;
        this.inertiaInv = 1 / this.inertia;
        
        this.colour = colour;
        this.type = "Particle";
    }

    setStatic() {
        this.mass = Infinity;
        this.massInv = 0;
        this.vel = VectorMath2.zero();
    }

    /**
     * @param {any} dt
     * @param {Vector2} gravity
     */
    simulate(dt, gravity) {
        //var kr = 0.01; //rolling friction canstant
        
        this.vel.addEqual(gravity.times(dt));
        if (this.mass === Infinity)
        {
            this.vel = VectorMath2.zero();
            //console.log(this.vel);
        }

        this.pos.addEqual(this.vel.times(dt));
        this.angularVel = 0;//particle doesn't rotate

        //this.angularVel *= (1 - kr);

        // if (this.vel.length() < eps)
        //     this.vel = VectorMath2.zero();
    }

    draw()
    {
        Draw.drawCircle(this.pos,this.radius,this.colour);
    }
}