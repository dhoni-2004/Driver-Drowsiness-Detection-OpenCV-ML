// This file exports TypeScript interfaces and types used across the application. 

export interface User {
    id: number;
    name: string;
    email: string;
}

export interface Product {
    id: number;
    name: string;
    price: number;
    description?: string;
}

export type ApiResponse<T> = {
    data: T;
    error?: string;
};

export type Nullable<T> = T | null;