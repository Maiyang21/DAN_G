/**
 * Simple test to verify frontend build configuration
 */

const fs = require('fs');
const path = require('path');

console.log('DAN_G Frontend Build Test');
console.log('=' .repeat(40));

// Test 1: Check if package.json exists and has required dependencies
console.log('\n1. Checking package.json...');
try {
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    const requiredDeps = [
        'react', 'next', 'react-bootstrap', 'react-icons', 
        'react-plotly.js', 'react-hot-toast', 'next-auth'
    ];
    
    const missingDeps = requiredDeps.filter(dep => !packageJson.dependencies[dep]);
    
    if (missingDeps.length === 0) {
        console.log('✅ All required dependencies found');
    } else {
        console.log(`❌ Missing dependencies: ${missingDeps.join(', ')}`);
    }
} catch (error) {
    console.log(`❌ Error reading package.json: ${error.message}`);
}

// Test 2: Check if TypeScript config exists
console.log('\n2. Checking TypeScript configuration...');
if (fs.existsSync('tsconfig.json')) {
    console.log('✅ tsconfig.json found');
} else {
    console.log('❌ tsconfig.json missing');
}

// Test 3: Check if Next.js config exists
console.log('\n3. Checking Next.js configuration...');
if (fs.existsSync('next.config.js')) {
    console.log('✅ next.config.js found');
} else {
    console.log('❌ next.config.js missing');
}

// Test 4: Check if main pages exist
console.log('\n4. Checking main pages...');
const requiredPages = [
    'pages/index.tsx',
    'pages/dashboard.tsx',
    'pages/api/auth/[...nextauth].ts'
];

requiredPages.forEach(page => {
    if (fs.existsSync(page)) {
        console.log(`✅ ${page} found`);
    } else {
        console.log(`❌ ${page} missing`);
    }
});

// Test 5: Check if type declarations exist
console.log('\n5. Checking type declarations...');
if (fs.existsSync('types/global.d.ts')) {
    console.log('✅ Type declarations found');
} else {
    console.log('❌ Type declarations missing');
}

console.log('\n' + '=' .repeat(40));
console.log('Frontend build test completed!');
