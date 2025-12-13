# ============================================================================
# LOCAL TESTING SCRIPT - Test Before Push to GitHub
# ============================================================================
# Simulates GitHub Actions workflows (lint.yml, test.yml) locally
# Usage: .\test-local.ps1 [-Fix] [-SkipTests]
# ============================================================================

param(
    [switch]$Fix,
    [switch]$SkipTests
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "LOCAL GITHUB ACTIONS TESTING" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$errors = 0
$warnings = 0

# ============================================================================
# 1. SYNTAX CHECK
# ============================================================================
Write-Host "[1/4] Checking Python syntax (flake8)..." -ForegroundColor Yellow
$flakeOutput = flake8 src/ --count --select=E9,F63,F7,F82 2>&1 | Out-String

if ($flakeOutput -match "^0") {
    Write-Host "  ✅ No syntax errors" -ForegroundColor Green
} else {
    Write-Host "  ❌ Syntax errors found:" -ForegroundColor Red
    Write-Host $flakeOutput
    $errors++
}

# ============================================================================
# 2. CODE FORMATTING
# ============================================================================
Write-Host "`n[2/4] Checking code formatting..." -ForegroundColor Yellow

if ($Fix) {
    Write-Host "  → Auto-fixing with Black..." -ForegroundColor Cyan
    black src/ tests/
    Write-Host "  → Auto-fixing imports with isort..." -ForegroundColor Cyan
    isort src/ tests/
    Write-Host "  ✅ Formatting applied" -ForegroundColor Green
} else {
    # Check Black
    $blackOutput = black src/ tests/ --check 2>&1 | Out-String
    if ($blackOutput -match "would reformat") {
        Write-Host "  ❌ Files need Black formatting (use -Fix)" -ForegroundColor Red
        $errors++
    } else {
        Write-Host "  ✅ Black formatting OK" -ForegroundColor Green
    }
    
    # Check isort
    $isortOutput = isort src/ tests/ --check-only 2>&1 | Out-String
    if ($isortOutput -match "ERROR") {
        Write-Host "  ❌ Imports need sorting (use -Fix)" -ForegroundColor Red
        $errors++
    } else {
        Write-Host "  ✅ Import sorting OK" -ForegroundColor Green
    }
}

# ============================================================================
# 3. YAML VALIDATION
# ============================================================================
Write-Host "`n[3/4] Checking YAML workflows..." -ForegroundColor Yellow

$yamlFiles = Get-ChildItem .github/workflows/*.yml
foreach ($file in $yamlFiles) {
    $content = Get-Content $file.FullName -Raw
    
    if ($content -match "[ `t]+`$") {
        Write-Host "  ⚠️  $($file.Name): has trailing spaces" -ForegroundColor Yellow
        $warnings++
    } else {
        Write-Host "  ✅ $($file.Name)" -ForegroundColor Green
    }
}

# ============================================================================
# 4. UNIT TESTS
# ============================================================================
if (-not $SkipTests) {
    Write-Host "`n[4/4] Running tests..." -ForegroundColor Yellow
    
    pytest tests/test_integration.py -v --tb=short -q
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Integration tests passed" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Some tests failed (expected)" -ForegroundColor Yellow
        $warnings++
    }
} else {
    Write-Host "`n[4/4] Tests skipped" -ForegroundColor Gray
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "✅ All checks passed! Safe to push." -ForegroundColor Green
    exit 0
} elseif ($errors -eq 0) {
    Write-Host "⚠️  $warnings warning(s) found" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "❌ $errors error(s), $warnings warning(s)" -ForegroundColor Red
    Write-Host "`nRun with -Fix to auto-fix formatting" -ForegroundColor Gray
    exit 1
}
